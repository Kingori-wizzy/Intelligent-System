import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import json

load_dotenv()

class ReasoningRAG:
    def __init__(self, db_dir="./chroma_db_storage"):
        """Initialize with existing vector database."""
        self.embeddings = OllamaEmbeddings(model="llama3.1")
        self.vector_db = Chroma(
            persist_directory=db_dir, 
            embedding_function=self.embeddings
        )
        self.llm = Ollama(model="llama3.1", temperature=0.3)  # Slightly higher temp for creativity
        
    def analyze_person_for_role(self, person_name, role_description):
        """Multi-step reasoning analysis."""
        
        print(f"\n🔍 ANALYZING: {person_name} for role: {role_description}")
        print("=" * 60)
        
        # STEP 1: Decompose the question into search queries
        print("\n📋 STEP 1: Identifying what to look for...")
        queries = self._generate_search_queries(person_name, role_description)
        
        # STEP 2: Retrieve relevant information for each aspect
        print("\n📚 STEP 2: Gathering evidence...")
        all_evidence = []
        for query in queries:
            docs = self.vector_db.similarity_search(query, k=2)
            all_evidence.extend(docs)
            print(f"   ✓ Retrieved for: {query[:50]}...")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_evidence = []
        for doc in all_evidence:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_evidence.append(doc)
        
        # STEP 3: Extract facts about the person
        print("\n📊 STEP 3: Extracting candidate profile...")
        profile = self._extract_profile(person_name, unique_evidence)
        print(json.dumps(profile, indent=2))
        
        # STEP 4: Analyze fit for role
        print("\n🧠 STEP 4: Reasoning about fit...")
        analysis = self._reason_about_fit(profile, role_description)
        
        # STEP 5: Generate final assessment
        print("\n✅ STEP 5: Generating final assessment...")
        final_assessment = self._generate_assessment(profile, role_description, analysis)
        
        return final_assessment
    
    def _generate_search_queries(self, person_name, role_description):
        """Generate targeted search queries based on role requirements."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at decomposing job requirements into search queries. Generate 4 specific aspects to look for in a candidate's profile."),
            ("human", f"""For the role: "{role_description}"
            
What 4 key aspects should we investigate about {person_name}?
Generate specific search queries that would find relevant evidence.

Format as a JSON list of strings:
["query1", "query2", "query3", "query4"]""")
        ])
        
        response = self.llm.invoke(prompt.format())
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
            else:
                queries = [
                    f"{person_name} education background",
                    f"{person_name} work experience",
                    f"{person_name} skills competencies",
                    f"{person_name} achievements projects"
                ]
        except:
            queries = [
                f"{person_name} education background",
                f"{person_name} work experience", 
                f"{person_name} skills competencies",
                f"{person_name} achievements projects"
            ]
        
        return queries[:4]
    
    def _extract_profile(self, person_name, documents):
        """Extract structured profile from documents."""
        context = "\n\n".join([doc.page_content for doc in documents[:5]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract factual information about the person. Only include information explicitly stated."),
            ("human", f"""Based on these documents, extract a structured profile of {person_name}:

{context}

Return as JSON with these fields (use null if not found):
{{
    "education": [],
    "work_experience": [],
    "skills": [],
    "achievements": [],
    "personal_attributes": []
}}
""")
        ])
        
        response = self.llm.invoke(prompt.format())
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                profile = json.loads(json_match.group())
            else:
                profile = {
                    "education": [],
                    "work_experience": [],
                    "skills": [],
                    "achievements": [],
                    "personal_attributes": []
                }
        except:
            profile = {
                "education": [],
                "work_experience": [],
                "skills": [],
                "achievements": [],
                "personal_attributes": []
            }
        
        return profile
    
    def _reason_about_fit(self, profile, role_description):
        """Reason about how profile matches role requirements."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a talent assessment expert. Analyze how a candidate's profile matches role requirements."),
            ("human", f"""CANDIDATE PROFILE:
Education: {profile.get('education', [])}
Work Experience: {profile.get('work_experience', [])}
Skills: {profile.get('skills', [])}
Achievements: {profile.get('achievements', [])}
Attributes: {profile.get('personal_attributes', [])}

ROLE: {role_description}

Provide a structured analysis:
1. STRONG MATCHES: What aspects of their background directly align?
2. TRANSFERABLE SKILLS: What could apply with some adaptation?
3. GAPS: What's missing from their profile?
4. POTENTIAL: What suggests they could learn/grow into this role?
""")
        ])
        
        return self.llm.invoke(prompt.format())
    
    def _generate_assessment(self, profile, role_description, analysis):
        """Generate final reasoned assessment."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior hiring manager providing a final assessment. Be evidence-based and nuanced."),
            ("human", f"""Based on this complete analysis:

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

ROLE: {role_description}

DETAILED ANALYSIS:
{analysis}

Provide a final assessment with:
1. Overall recommendation (Strong Yes / Yes with reservations / Not enough info / No)
2. Key evidence supporting this recommendation
3. Specific areas where candidate exceeds requirements
4. Specific areas where candidate falls short
5. Recommended development areas if hired
6. Confidence level in this assessment (High/Medium/Low)

Format as clear sections with emoji headers.
""")
        ])
        
        return self.llm.invoke(prompt.format())

def main():
    print("=" * 60)
    print("🧠 REASONING RAG - TALENT ANALYST")
    print("=" * 60)
    
    # Initialize
    rag = ReasoningRAG()
    
    # Example analysis
    person = "Sharleen"
    role = "Psychologist - need to understand human behavior, provide counseling, conduct assessments, and communicate empathetically"
    
    assessment = rag.analyze_person_for_role(person, role)
    
    print("\n" + "=" * 60)
    print("📋 FINAL ASSESSMENT")
    print("=" * 60)
    print(assessment)

if __name__ == "__main__":
    main()