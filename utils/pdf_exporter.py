"""
PDF Export Module
Converts generated medical reports to professional PDF format with visualizations
"""
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import io
import tempfile

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: reportlab not installed. PDF export disabled.")
    REPORTLAB_AVAILABLE = False

try:
    import markdown
    from markdown.extensions import tables, fenced_code
    MARKDOWN_AVAILABLE = True
except ImportError:
    print("Warning: markdown not installed. Markdown to PDF conversion limited.")
    MARKDOWN_AVAILABLE = False


class PDFExporter:
    """Exports medical reports to PDF format"""
    
    def __init__(self):
        """Initialize PDF exporter"""
        self.page_size = letter
        self.styles = None
        
        if REPORTLAB_AVAILABLE:
            self._initialize_styles()
    
    def _initialize_styles(self):
        """Initialize PDF styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom styles - check if they exist before adding
        custom_styles = {
            'CustomTitle': {
                'parent': self.styles['Heading1'],
                'fontSize': 24,
                'textColor': colors.HexColor('#1f77b4'),
                'spaceAfter': 30,
                'alignment': TA_CENTER,
                'fontName': 'Helvetica-Bold'
            },
            'SectionHeader': {
                'parent': self.styles['Heading2'],
                'fontSize': 16,
                'textColor': colors.HexColor('#1f77b4'),
                'spaceAfter': 12,
                'spaceBefore': 20,
                'fontName': 'Helvetica-Bold'
            },
            'SubHeader': {
                'parent': self.styles['Heading3'],
                'fontSize': 14,
                'textColor': colors.HexColor('#333333'),
                'spaceAfter': 10,
                'spaceBefore': 15,
                'fontName': 'Helvetica-Bold'
            },
            'BodyText': {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'textColor': colors.black,
                'alignment': TA_JUSTIFY,
                'spaceAfter': 10,
                'leading': 14
            },
            'Citation': {
                'parent': self.styles['Normal'],
                'fontSize': 9,
                'textColor': colors.HexColor('#666666'),
                'leftIndent': 20,
                'italic': True
            }
        }
        
        # Add custom styles only if they don't exist
        for style_name, style_props in custom_styles.items():
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(name=style_name, **style_props))
    
    def _add_chart_image(self, fig, story: List, caption: str = ""):
        """
        Add a plotly chart as an image to the PDF
        
        Args:
            fig: Plotly figure object
            story: List to append PDF elements to
            caption: Chart caption text
        """
        try:
            # Convert plotly figure to image
            img_bytes = fig.to_image(format="png", width=600, height=400)
            img_buffer = io.BytesIO(img_bytes)
            
            # Create ReportLab Image
            img = RLImage(img_buffer, width=5.5*inch, height=3.5*inch)
            story.append(img)
            
            if caption:
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(f"<i>{caption}</i>", self.styles['Citation']))
            
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Error adding chart to PDF: {str(e)}")
            story.append(Paragraph(f"[Chart: {caption}]", self.styles['BodyText']))
    
    def export_report_to_pdf(self, report: Dict[str, Any], 
                             output_path: str,
                             chart_images: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export report to PDF
        
        Args:
            report: Report dictionary
            output_path: Path to save PDF
            
        Returns:
            Result with success status and file path
        """
        if not REPORTLAB_AVAILABLE:
            return {
                "success": False,
                "error": "reportlab not installed. Install with: pip install reportlab"
            }
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.page_size,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content
            story = []
            
            # Add title
            story.append(Paragraph(report.get('title', 'Medical Report'), 
                                 self.styles['CustomTitle']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add source document at the top (only once)
            sources = report.get('sources_used', [])
            if sources:
                source_name = sources[0] if len(sources) == 1 else ', '.join(sources)
                source_text = f"<b>Source:</b> {source_name}"
                story.append(Paragraph(source_text, self.styles['BodyText']))
                story.append(Spacer(1, 0.3*inch))
            
            # Add horizontal line
            story.append(self._create_horizontal_line())
            story.append(Spacer(1, 0.2*inch))
            
            # Add sections
            sections = report.get('generated_sections', {})
            chart_index = 0
            
            for section_name, section_data in sections.items():
                # Section content
                content = section_data.get('content', '')
                
                # Skip sections with no content or generic "not found" messages
                if not content or content.strip() in [
                    "No relevant information found in the selected documents.",
                    "No information available for this section.",
                    "*No information available for this section.*"
                ]:
                    continue
                
                # Section title
                title = self._format_section_title(section_name)
                story.append(Paragraph(title, self.styles['SectionHeader']))
                story.append(Spacer(1, 0.1*inch))
                
                # Add charts for graphs_and_charts section
                if section_name == "graphs_and_charts" and chart_images and len(chart_images) > 0:
                    # Add all collected charts
                    for chart_type, fig, chart_title in chart_images:
                        try:
                            # Add chart title
                            story.append(Paragraph(f"<b>{chart_title}</b>", self.styles['SubHeader']))
                            story.append(Spacer(1, 0.1*inch))
                            
                            # Add chart image
                            self._add_chart_image(fig, story, caption=chart_title)
                            
                            chart_index += 1
                        except Exception as e:
                            print(f"Error adding chart '{chart_title}' to PDF: {str(e)}")
                            story.append(Paragraph(f"[Chart: {chart_title}]", self.styles['BodyText']))
                            story.append(Spacer(1, 0.2*inch))
                    
                    # Skip adding text content since we've added charts
                    continue
                
                # Add tables for patient_tables section
                if section_name == "patient_tables" and content:
                    # Extract and format tables from content
                    import re
                    import pandas as pd
                    
                    # Look for markdown tables
                    table_pattern = r'\|([^\n]+)\|\n\|[-:\s|]+\|\n((?:\|[^\n]+\|\n)+)'
                    table_matches = re.findall(table_pattern, content)
                    
                    for table_match in table_matches:
                        header_line = table_match[0]
                        data_lines = table_match[1]
                        
                        # Parse header
                        headers = [h.strip() for h in header_line.split('|') if h.strip()]
                        
                        # Parse data rows
                        data_rows = []
                        for line in data_lines.split('\n'):
                            if line.strip() and '|' in line:
                                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                if row and len(row) == len(headers):
                                    data_rows.append(row)
                        
                        if headers and data_rows:
                            # Create ReportLab table
                            table_data = [headers] + data_rows
                            pdf_table = Table(table_data, repeatRows=1)
                            pdf_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                ('FONTSIZE', (0, 1), (-1, -1), 9),
                                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                            ]))
                            story.append(pdf_table)
                            story.append(Spacer(1, 0.2*inch))
                
                if content:
                    # Split into paragraphs
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            # Check if it's a heading
                            if para.strip().startswith('###'):
                                heading_text = para.strip().replace('###', '').strip()
                                # Clean heading text too
                                heading_text = self._clean_text_for_pdf(heading_text)
                                story.append(Paragraph(heading_text, self.styles['SubHeader']))
                            elif para.strip().startswith('##'):
                                heading_text = para.strip().replace('##', '').strip()
                                # Clean heading text too
                                heading_text = self._clean_text_for_pdf(heading_text)
                                story.append(Paragraph(heading_text, self.styles['SectionHeader']))
                            else:
                                # Clean and add paragraph
                                clean_text = self._clean_text_for_pdf(para.strip())
                                story.append(Paragraph(clean_text, self.styles['BodyText']))
                    
                    story.append(Spacer(1, 0.1*inch))
                
                # Add tables if present
                tables = section_data.get('tables', [])
                if tables:
                    story.append(Paragraph("Data Tables:", self.styles['SubHeader']))
                    for table in tables:
                        story.append(self._create_table_from_text(table.get('content', '')))
                        story.append(Spacer(1, 0.1*inch))
                
                # Add chart references
                chart_refs = section_data.get('chart_refs', [])
                if chart_refs:
                    story.append(Paragraph("Referenced Figures:", self.styles['SubHeader']))
                    for ref in chart_refs:
                        ref_text = f"• {ref.get('reference', '')}"
                        story.append(Paragraph(ref_text, self.styles['BodyText']))
                    story.append(Spacer(1, 0.1*inch))
                
                # Add summary if present
                summary = section_data.get('summary', '')
                if summary:
                    story.append(Paragraph("Summary:", self.styles['SubHeader']))
                    clean_summary = self._clean_text_for_pdf(summary)
                    story.append(Paragraph(clean_summary, self.styles['BodyText']))
                    story.append(Spacer(1, 0.1*inch))
                
                # Don't add sources after each section (already shown at top)
                
                story.append(Spacer(1, 0.3*inch))
            
            # Don't add references section at the bottom (source already shown at top)
            
            # Build PDF
            doc.build(story)
            
            return {
                "success": True,
                "file_path": output_path,
                "message": f"PDF exported successfully to {output_path}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to export PDF: {str(e)}"
            }
    
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF rendering"""
        import re
        
        # First escape HTML special characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Handle markdown-style bold (**text**)
        # Replace pairs of ** with <b> and </b>
        def replace_bold(match):
            return f'<b>{match.group(1)}</b>'
        text = re.sub(r'\*\*(.+?)\*\*', replace_bold, text)
        
        # Handle markdown-style italic (*text*)
        # Only replace single * that aren't part of **
        def replace_italic(match):
            return f'<i>{match.group(1)}</i>'
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', replace_italic, text)
        
        # Handle bullet points
        text = re.sub(r'^\s*[-•]\s+', '• ', text, flags=re.MULTILINE)
        
        # Remove any remaining single asterisks
        text = text.replace('*', '')
        
        return text
    
    def _format_section_title(self, section_name: str) -> str:
        """Format section name for display"""
        titles = {
            "introduction": "Introduction",
            "clinical_findings": "Clinical Findings",
            "patient_tables": "Patient Data Tables",
            "graphs_and_charts": "Graphs and Charts",
            "diagnosis": "Diagnosis",
            "treatment_plan": "Treatment Plan",
            "summary": "Summary"
        }
        return titles.get(section_name, section_name.replace('_', ' ').title())
    
    def _create_horizontal_line(self):
        """Create a horizontal line separator"""
        line_data = [['_' * 100]]
        table = Table(line_data, colWidths=[6.5*inch])
        table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f77b4')),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        return table
    
    def _create_table_from_text(self, table_text: str):
        """Convert text table to PDF table"""
        lines = table_text.strip().split('\n')
        
        # Parse table rows
        table_data = []
        for line in lines:
            if '|' in line:
                # Markdown-style table
                cells = [cell.strip() for cell in line.split('|')]
                cells = [c for c in cells if c]  # Remove empty cells
                if cells:
                    table_data.append(cells)
            elif '\t' in line:
                # Tab-separated
                cells = [cell.strip() for cell in line.split('\t')]
                if cells:
                    table_data.append(cells)
        
        if not table_data:
            return Paragraph(table_text, self.styles['BodyText'])
        
        # Create PDF table
        try:
            # Calculate column widths
            num_cols = max(len(row) for row in table_data)
            col_width = 6.5 * inch / num_cols
            
            # Ensure all rows have same number of columns
            for row in table_data:
                while len(row) < num_cols:
                    row.append('')
            
            table = Table(table_data, colWidths=[col_width] * num_cols)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            
            return table
        
        except Exception as e:
            # Fallback to plain text
            return Paragraph(table_text, self.styles['BodyText'])
    
    def export_markdown_to_pdf(self, markdown_text: str, 
                              output_path: str) -> Dict[str, Any]:
        """
        Export markdown text directly to PDF
        
        Args:
            markdown_text: Markdown formatted text
            output_path: Path to save PDF
            
        Returns:
            Result dictionary
        """
        if not REPORTLAB_AVAILABLE:
            return {
                "success": False,
                "error": "reportlab not installed"
            }
        
        try:
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
            story = []
            
            # Parse markdown sections
            sections = markdown_text.split('\n## ')
            
            # First section is title
            if sections:
                title_section = sections[0]
                title_lines = title_section.split('\n')
                title = title_lines[0].replace('#', '').strip()
                
                story.append(Paragraph(title, self.styles['CustomTitle']))
                story.append(Spacer(1, 0.3*inch))
                
                # Process remaining sections
                for section in sections[1:]:
                    lines = section.split('\n')
                    section_title = lines[0].strip()
                    
                    story.append(Paragraph(section_title, self.styles['SectionHeader']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    # Process content
                    content = '\n'.join(lines[1:])
                    paragraphs = content.split('\n\n')
                    
                    for para in paragraphs:
                        if para.strip():
                            clean_text = self._clean_text_for_pdf(para.strip())
                            story.append(Paragraph(clean_text, self.styles['BodyText']))
                    
                    story.append(Spacer(1, 0.2*inch))
            
            # Build PDF
            doc.build(story)
            
            return {
                "success": True,
                "file_path": output_path
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
