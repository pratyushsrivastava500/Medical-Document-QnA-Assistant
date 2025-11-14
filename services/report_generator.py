"""
Report Generation Module
Handles structured medical report generation from uploaded documents
"""
from typing import List, Dict, Any, Optional
import json
import re
import os
from services.llm_client import LLMClient
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.embeddings import EmbeddingGenerator
import streamlit as st


class ReportGenerator:
    """Generates structured medical reports using LLM function calling"""
    
    # Available report sections
    AVAILABLE_SECTIONS = [
        "introduction",
        "clinical_findings",
        "patient_tables",
        "diagnosis",
        "treatment_plan",
        "summary"
    ]
    
    def __init__(self, llm_client: LLMClient, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 document_processor: DocumentProcessor):
        """
        Initialize Report Generator
        
        Args:
            llm_client: LLM client for generation
            vector_store: Vector store for document retrieval
            embedding_generator: Embedding generator
            document_processor: Document processor for extraction
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.document_processor = document_processor
    
    def generate_report(self, 
                       sections: List[str],
                       user_instructions: Optional[str] = None,
                       selected_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a structured medical report
        
        Args:
            sections: List of sections to include in the report
            user_instructions: Additional user instructions for the report
            selected_sources: Optional list of specific source files to use (None = use all)
            
        Returns:
            Dictionary containing the complete report with all sections
        """
        # Validate sections
        invalid_sections = [s for s in sections if s not in self.AVAILABLE_SECTIONS]
        if invalid_sections:
            return {
                "success": False,
                "error": f"Invalid sections: {', '.join(invalid_sections)}",
                "available_sections": self.AVAILABLE_SECTIONS
            }
        
        # Check if documents are available
        if self.vector_store.count() == 0:
            return {
                "success": False,
                "error": "No documents available. Please upload documents first."
            }
        
        # Validate selected sources if provided
        all_sources = self.vector_store.get_all_sources()
        if selected_sources:
            invalid_sources = [s for s in selected_sources if s not in all_sources]
            if invalid_sources:
                return {
                    "success": False,
                    "error": f"Invalid source files: {', '.join(invalid_sources)}",
                    "available_sources": all_sources
                }
            sources_to_use = selected_sources
        else:
            sources_to_use = all_sources
        
        # Generate report structure
        report = {
            "success": True,
            "title": "Medical Document Report",
            "generated_sections": {},
            "sources_used": [],
            "metadata": {
                "total_documents": len(sources_to_use),
                "all_available_documents": len(all_sources),
                "selected_documents": sources_to_use,
                "sections_requested": sections,
                "user_instructions": user_instructions
            }
        }
        
        # Generate each requested section (except summary)
        regular_sections = [s for s in sections if s != "summary"]
        for section in regular_sections:
            section_data = self._generate_section(section, user_instructions, sources_to_use)
            report["generated_sections"][section] = section_data
            
            # Collect sources
            if "sources" in section_data:
                report["sources_used"].extend(section_data["sources"])
        
        # Generate summary if requested (based on all other sections)
        if "summary" in sections:
            summary_data = self._generate_overall_summary(report["generated_sections"], sources_to_use)
            report["generated_sections"]["summary"] = summary_data
            if "sources" in summary_data:
                report["sources_used"].extend(summary_data["sources"])
        
        # Remove duplicate sources and filter by selected
        report["sources_used"] = [s for s in list(set(report["sources_used"])) 
                                  if not selected_sources or s in selected_sources]
        
        return report
    
    def _generate_section(self, section_name: str, 
                         user_instructions: Optional[str] = None,
                         selected_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a specific section of the report
        
        Args:
            section_name: Name of the section to generate
            user_instructions: User instructions
            selected_sources: Optional list of sources to filter results
            
        Returns:
            Dictionary with section content and metadata
        """
        # Define section-specific queries
        section_queries = {
            "introduction": "introduction, background, patient information, overview",
            "clinical_findings": "clinical findings, observations, examination results, test results",
            "patient_tables": "patient data tables, measurements, lab results, vitals",
            "diagnosis": "diagnosis, medical conditions, diseases, disorders identified",
            "treatment_plan": "treatment, therapy, medication, intervention, care plan",
            "summary": "summary, conclusion, recommendations, follow-up"
        }
        
        query = section_queries.get(section_name, section_name)
        
        # Use function calling to determine extraction method
        extraction_result = self._extract_section_content(section_name, query, user_instructions, selected_sources)
        
        return extraction_result
    
    def _extract_section_content(self, section_name: str, query: str,
                                 user_instructions: Optional[str] = None,
                                 selected_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract content for a section using LLM function calling
        
        Args:
            section_name: Section name
            query: Search query
            user_instructions: User instructions
            selected_sources: Optional list of sources to filter results
            
        Returns:
            Extracted content with metadata
        """
        # Retrieve relevant documents
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        results = self.vector_store.search(query_embedding, top_k=10)
        
        # Filter results by selected sources if specified
        if selected_sources:
            results = [r for r in results if r["metadata"].get("source") in selected_sources]
        
        if not results:
            return {
                "section_name": section_name,
                "content": "No relevant information found in the selected documents.",
                "extraction_type": "none",
                "sources": []
            }
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for idx, result in enumerate(results, 1):
            content = result["content"]
            metadata = result["metadata"]
            score = result["score"]
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"--- Document Chunk {idx} (from {source}) ---\n{content}\n")
            sources.append(source)
        
        context = "\n".join(context_parts)
        
        # Determine if section needs summary or exact extraction
        needs_summary = user_instructions and "summary" in user_instructions.lower()
        
        # Define function calling tools
        tools = self._get_extraction_tools()
        
        # Create enhanced prompt for professional report writing
        system_prompt = self._create_report_writing_prompt(section_name, needs_summary)
        
        # Load standard section prompt
        section_prompt_path = os.path.join('prompts', 'report_section_user_prompt.txt')
        try:
            with open(section_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt = f.read().format(
                    section_name=section_name.replace('_', ' ').title(),
                    query=query,
                    context=context,
                    user_instructions=user_instructions if user_instructions else "Write clear, professional medical content"
                )
        except Exception as e:
            # Fallback to hardcoded prompt
            user_prompt = f"""Write the **{section_name.replace('_', ' ').title()}** section for a professional medical analysis report.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

You are a medical report writer. Your job is to READ the document chunks below and WRITE a professional, well-formatted section. 

DO NOT copy-paste raw text chunks!
DO NOT include partial sentences or broken words!
DO NOT include page numbers, footers, or "Downloaded from" text!

INSTEAD:
- Read and understand the medical information
- Synthesize it into clear, complete sentences
- Organize into logical paragraphs
- Use medical terminology appropriately
- Present findings in a structured way

**Section Topic:** {query}

**Source Material to Analyze:**
{context}

**Writing Guidelines:**
1. Start with the most relevant information for this section
2. Write in complete, professional sentences
3. Use bullet points for lists of symptoms, findings, or recommendations
4. Include specific medical values, dates, or measurements when present
5. Maintain professional medical report tone
6. Organize content logically (general → specific, or by importance)

**User Requirements:** {user_instructions if user_instructions else "Write clear, professional medical content"}

NOW WRITE THIS SECTION AS A PROFESSIONAL MEDICAL REPORT - not as raw document chunks!
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM with function calling
        response = self._call_llm_with_functions(messages, tools)

        # Clean extracted text to remove hyphenation, page footers, and bad line breaks
        extracted = response.get("extracted_content", "")
        summary = response.get("summary", None)

        cleaned_extracted = self._clean_text(extracted)
        # Remove section heading if LLM added it
        cleaned_extracted = self._remove_duplicate_heading(cleaned_extracted, section_name)
        cleaned_summary = self._clean_text(summary) if summary else None

        return {
            "section_name": section_name,
            "content": cleaned_extracted,
            "summary": cleaned_summary,
            "extraction_type": response.get("extraction_type", "text"),
            "tables": response.get("tables", []),
            "sources": list(set(sources))
        }
    
    def _remove_duplicate_heading(self, text: str, section_name: str) -> str:
        """
        Remove duplicate section headings from the beginning of the content
        
        Args:
            text: Content text
            section_name: Name of the section
            
        Returns:
            Text with duplicate headings removed
        """
        if not text:
            return text
        
        # Convert section name to various heading formats
        section_title = section_name.replace('_', ' ').title()
        
        # Remove heading patterns from the start of text
        patterns = [
            rf'^#+\s*{re.escape(section_title)}\s*\n+',  # Markdown headings
            rf'^\*\*{re.escape(section_title)}\*\*\s*\n+',  # Bold headings
            rf'^{re.escape(section_title)}:\s*\n+',  # Colon format
            rf'^{re.escape(section_title)}\s*\n+',  # Plain text heading
            rf'^#+\s*{re.escape(section_title.upper())}\s*\n+',  # Uppercase markdown
            rf'^\*\*{re.escape(section_title.upper())}\*\*\s*\n+',  # Uppercase bold
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text.strip()

    def _clean_text(self, text: Optional[str]) -> str:
        """
        Clean extracted text by removing hyphenation at line breaks, common
        page footers/headers, and normalizing line breaks into paragraphs.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove common PDF/HTML footers, headers, and metadata
        patterns = [
            r"Downloaded from .*?\d{4}",  # Downloaded from URL date
            r"--- Page \d+ ---",
            r"Page \d+",
            r"SPECTRUM\.[A-Z\.]+",
            r"Downloaded by .* on .*",
            r"\d{2,4} ADA HEALTH CARE.*?ADDRESS",
            r"Accessed \d{1,2} [A-Za-z]+ \d{4}",
            r"Available from www\..*",
            r"Figure \d+ is a screen shot.*",
            r"technology platform, and a social determinants.*",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)

        # Fix hyphenation at line breaks: word-\nnext -> wordnext
        text = re.sub(r"(\w+)\s*-\s*\n+\s*(\w+)", r"\1\2", text)
        
        # Remove standalone fragments that look like broken words at line start
        text = re.sub(r"^\s*ing\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*tion\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*nique\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*cated\s+", "", text, flags=re.MULTILINE)

        # Normalize remaining single line breaks inside paragraphs
        # Split into paragraphs by double newlines, collapse single newlines
        paragraphs = re.split(r"\n\s*\n+", text)
        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove leading/trailing whitespace and collapse internal newlines into spaces
            p = re.sub(r"\s*\n\s*", " ", para).strip()
            # Remove stray multiple spaces
            p = re.sub(r" {2,}", " ", p)
            # Remove very short fragments (likely artifacts)
            if p and len(p) > 15:
                cleaned_paragraphs.append(p)

        return "\n\n".join(cleaned_paragraphs).strip()
    
    def _generate_overall_summary(self, generated_sections: Dict[str, Any], 
                                  selected_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate an overall summary based on all generated sections
        
        Args:
            generated_sections: Dictionary of already generated sections
            selected_sources: Optional list of sources to use
            
        Returns:
            Summary section data
        """
        # Collect all content from generated sections
        all_content = []
        all_sources = set()
        
        section_titles = {
            "introduction": "Introduction",
            "clinical_findings": "Clinical Findings",
            "patient_tables": "Patient Data",
            "diagnosis": "Diagnosis",
            "treatment_plan": "Treatment Plan"
        }
        
        for section_name, section_data in generated_sections.items():
            if section_name == "summary":
                continue  # Skip if somehow summary is already there
            
            content = section_data.get("content", "").strip()
            if content and content != "No relevant information found in the selected documents.":
                title = section_titles.get(section_name, section_name.replace("_", " ").title())
                all_content.append(f"{title}:\n{content}")
                
                # Collect sources
                if "sources" in section_data:
                    all_sources.update(section_data["sources"])
        
        if not all_content:
            return {
                "section_name": "summary",
                "content": "Unable to generate summary as no relevant content was found in the document.",
                "extraction_type": "text",
                "sources": []
            }
        
        # Create a prompt for LLM to summarize all sections
        combined_content = "\n\n".join(all_content)
        
        # Load prompts from files
        system_prompt_path = os.path.join('prompts', 'report_summary_system_prompt.txt')
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except Exception as e:
            # Fallback to hardcoded prompt
            system_prompt = """You are a medical report summarizer. Your task is to create a comprehensive executive summary of the medical report.

**Your summary should:**
1. Synthesize key findings from all sections
2. Highlight the most critical medical information
3. Present important diagnoses, findings, or recommendations
4. Be concise yet comprehensive (3-5 paragraphs)
5. Use professional medical language
6. Focus on actionable insights and important conclusions

**Do NOT:**
- Simply list what each section contains
- Include irrelevant details
- Mention technical report structure
- Use phrases like "this report contains" or "the document discusses"

Write a professional executive summary now."""

        user_prompt = f"""Based on the following medical report content, create a comprehensive executive summary:

{combined_content}

Provide a clear, professional summary that captures the essential medical information and key findings."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Generate summary using LLM
            summary_text = self.llm_client.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Clean the summary
            cleaned_summary = self._clean_text(summary_text)
            
            return {
                "section_name": "summary",
                "content": cleaned_summary,
                "extraction_type": "text",
                "sources": list(all_sources)
            }
            
        except Exception as e:
            return {
                "section_name": "summary",
                "content": f"Error generating summary: {str(e)}",
                "extraction_type": "error",
                "sources": list(all_sources)
            }
    
    def _get_extraction_tools(self) -> List[Dict[str, Any]]:
        """
        Define function calling tools for content extraction
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_exact_content",
                    "description": "Extract exact content from the documents without modification. Use this to preserve original text, findings, or data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extracted_content": {
                                "type": "string",
                                "description": "The exact content extracted from the documents, preserved as-is"
                            },
                            "extraction_type": {
                                "type": "string",
                                "enum": ["text", "table", "list", "mixed"],
                                "description": "Type of content extracted"
                            }
                        },
                        "required": ["extracted_content", "extraction_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_table_data",
                    "description": "Extract tabular data or patient data tables from the documents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "content": {"type": "string"},
                                        "source": {"type": "string"}
                                    }
                                },
                                "description": "Array of tables found in the documents"
                            }
                        },
                        "required": ["tables"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_and_summarize",
                    "description": "Extract exact content and also provide a summary when requested by user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extracted_content": {
                                "type": "string",
                                "description": "The exact content extracted from documents"
                            },
                            "summary": {
                                "type": "string",
                                "description": "A concise summary of the extracted content"
                            }
                        },
                        "required": ["extracted_content", "summary"]
                    }
                }
            }
        ]
    
    def _create_report_writing_prompt(self, section_name: str, 
                                      needs_summary: bool) -> str:
        """
        Create system prompt for professional report writing (not raw extraction)
        
        Args:
            section_name: Section name
            needs_summary: Whether summary is needed
            
        Returns:
            System prompt string for report synthesis
        """
        base_prompt = f"""You are an expert medical report writer. Your role is to analyze medical document content and write professional, well-structured report sections.

**CORE WRITING PRINCIPLES:**

1. **Synthesize, Don't Copy**: Read the source material and rewrite it into proper report prose. Never copy raw chunks verbatim.

2. **Professional Format**: Write in complete paragraphs with proper medical report structure. Use clear topic sentences and logical flow.

3. **Remove Artifacts**: Eliminate all PDF artifacts including:
   - Broken words with hyphens (infec-tion → infection)
   - Page numbers, headers, footers
   - "Downloaded from..." or citation text
   - Incomplete sentences or fragments

4. **Medical Clarity**: 
   - Use appropriate medical terminology
   - Include specific data (lab values, measurements, dates)
   - Present findings clearly and concisely
   - Maintain objective, professional tone

5. **Logical Organization**:
   - Most important/critical information first
   - Group related findings together
   - Use bullet points for lists (symptoms, findings, recommendations)
   - Use paragraphs for explanations and context

6. **Content Quality**:
   - Complete sentences only
   - No sentence fragments
   - No mid-word breaks
   - Proper paragraph structure
   - Professional medical language

**SECTION TYPE: {section_name.replace('_', ' ').title()}**

For this section, focus on writing content that would appear in a professional medical analysis report. The reader should see polished, publication-quality prose - not raw document excerpts.
"""
        
        if needs_summary:
            base_prompt += """

**SUMMARY REQUIREMENT:**
After writing the main content, provide a "Key Points" section with 3-5 bullet points highlighting the most important findings in simple, clear language.
"""
        
        base_prompt += """

**OUTPUT FORMAT:**
Provide your written content directly. Write as if you're authoring this section of the report from scratch, using the source documents as reference material.
"""
        
        return base_prompt
    
    def _create_extraction_system_prompt(self, section_name: str, 
                                        needs_summary: bool) -> str:
        """
        Create system prompt for extraction
        
        Args:
            section_name: Section name
            needs_summary: Whether summary is needed
            
        Returns:
            System prompt string
        """
        # Load base prompt from file
        prompt_path = os.path.join('prompts', 'report_section_system_prompt.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                base_prompt = f.read().format(section_name=section_name.replace('_', ' ').title())
        except Exception as e:
            # Fallback to hardcoded prompt if file not found
            base_prompt = f"""You are a professional medical report writer creating the "{section_name.replace('_', ' ').title()}" section of a formal medical document.

**CRITICAL INSTRUCTIONS:**

**DO NOT:**
- Copy raw text fragments with broken words or hyphenation
- Include page numbers, footers, headers, or citations
- Copy incomplete sentences or phrases
- Include "Downloaded from..." or technical document metadata
- Use phrases like "Document 1 says..." or "According to chunk..."

**DO:**
- Write complete, professional paragraphs in medical report style
- Synthesize information into clear, flowing prose
- Use proper medical terminology and formal language
- Present findings in logical order (most important first)
- Format lists with bullet points when presenting multiple items
- Include specific data (measurements, dates, values) when present
- Ensure every sentence is complete and grammatically correct

**WRITING STYLE:**
- Professional, third-person medical report tone
- Clear and concise (no flowery language)
- Factual and objective
- Properly structured paragraphs
- Use medical terminology appropriately

**STRUCTURE:**
- Start with main findings or overview
- Present specific details in logical order
- Use bullet points for multiple findings or recommendations
- End with conclusions or important notes if applicable
"""
        
        if needs_summary:
            base_prompt += """
**KEY POINTS SUMMARY:**
After the main section content, add a "Key Points" subsection with:
- 3-5 most important findings in bullet format
- Clear, concise statements
- Each point on its own line
"""
        
        return base_prompt
    
    def _call_llm_with_functions(self, messages: List[Dict[str, str]], 
                                 tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call LLM with function calling capability
        
        Args:
            messages: Conversation messages
            tools: Available tools
            
        Returns:
            Extracted content
        """
        try:
            # Note: Groq's function calling API
            # Since the exact API might vary, we'll implement a simpler version
            # that uses structured prompts instead of native function calling
            
            # Add tool descriptions to system message
            tool_descriptions = "\n\n".join([
                f"Function: {tool['function']['name']}\n"
                f"Description: {tool['function']['description']}\n"
                f"Parameters: {json.dumps(tool['function']['parameters'], indent=2)}"
                for tool in tools
            ])
            
            enhanced_system = messages[0]["content"]
            
            messages[0]["content"] = enhanced_system
            
            # Use higher temperature and more tokens for better report writing
            response = self.llm_client.generate_response(
                messages=messages,
                temperature=0.4,  # Higher for more natural writing
                max_tokens=4000   # More space for well-written content
            )
            
            # Parse response to extract structured data
            parsed = self._parse_extraction_response(response)
            
            return parsed
            
        except Exception as e:
            return {
                "extracted_content": f"Error during extraction: {str(e)}",
                "extraction_type": "error"
            }
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract structured data
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed structured data
        """
        # Simple parsing logic
        result = {
            "extracted_content": response,
            "extraction_type": "text"
        }
        
        # Check for summary markers
        if "SUMMARY:" in response.upper() or "Summary:" in response:
            parts = response.split("SUMMARY:" if "SUMMARY:" in response else "Summary:")
            if len(parts) == 2:
                result["extracted_content"] = parts[0].strip()
                result["summary"] = parts[1].strip()
        
        # Check for table markers
        if "TABLE" in response.upper() or "|" in response:
            result["extraction_type"] = "table"
        
        return result
    
    def format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """
        Format the report as a professional markdown document
        
        Args:
            report: Report dictionary
            
        Returns:
            Markdown formatted report with improved structure
        """
        if not report.get("success", False):
            return f"# Report Generation Failed\n\n**Error:** {report.get('error', 'Unknown error')}"
        
        # Professional header - just the title, no technical metadata
        md_parts = [
            "# Medical Analysis Report\n\n",
            "---\n\n"
        ]
        
        # Get metadata
        metadata = report.get('metadata', {})
        sources_list = report.get('sources_used', [])
        
        # Document information (professional format)
        if sources_list:
            source_name = sources_list[0] if len(sources_list) == 1 else "Multiple Documents"
            md_parts.append(f"**Document:** {source_name}\n\n")
        
        md_parts.append("---\n\n")
        
        # Add each section with improved formatting
        sections = report.get("generated_sections", {})
        
        section_titles = {
            "introduction": "Introduction",
            "clinical_findings": "Clinical Findings",
            "patient_tables": "Patient Data",
            "diagnosis": "Diagnosis",
            "treatment_plan": "Treatment Plan",
            "summary": "Summary"
        }
        
        for section_name, section_data in sections.items():
            title = section_titles.get(section_name, section_name.replace("_", " ").title())
            
            md_parts.append(f"## {title}\n\n")
            
            # Add main content with better formatting
            content = section_data.get("content", "").strip()
            if content:
                # Clean up content - remove excessive line breaks
                lines = content.split('\n')
                cleaned_lines = []
                prev_empty = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if not prev_empty:
                            cleaned_lines.append('')
                        prev_empty = True
                    else:
                        cleaned_lines.append(line)
                        prev_empty = False
                
                content = '\n\n'.join([l for l in cleaned_lines if l])
                md_parts.append(f"{content}\n\n")
            else:
                md_parts.append("*No information available for this section.*\n\n")
            
            # Add summary if exists (but label it professionally)
            summary = section_data.get("summary")
            if summary and summary.strip():
                md_parts.append(f"**Key Points:**\n\n{summary.strip()}\n\n")
            
            md_parts.append("---\n\n")
        
        # Footer - professional disclaimer
        md_parts.append("*This report is based on the analysis of provided medical documentation. ")
        md_parts.append("All information should be verified against original source materials.*\n")
        
        return "".join(md_parts)
