#!/usr/bin/env python3
"""
Interactive Knowledge Base Viewer
View processed documents from enlitens_knowledge_base.json.temp
"""

import json
import sys
from pathlib import Path

def print_separator(char="=", length=80):
    print(char * length)

def view_summary(kb):
    """Show overall summary"""
    print_separator()
    print("📊 KNOWLEDGE BASE SUMMARY")
    print_separator()
    print(f"Version: {kb.get('version', 'N/A')}")
    print(f"Created: {kb.get('created_at', 'N/A')}")
    print(f"Total Documents: {kb.get('total_documents', 0)}")
    print(f"Documents Processed: {len(kb.get('documents', []))}")
    print()

def view_document(doc, doc_num):
    """Show detailed view of a single document"""
    print_separator()
    print(f"📄 DOCUMENT #{doc_num}")
    print_separator()
    print()
    
    # Metadata
    if 'metadata' in doc:
        meta = doc['metadata']
        print("📋 METADATA:")
        print(f"   Document ID: {meta.get('document_id', 'N/A')}")
        print(f"   Filename: {meta.get('filename', 'N/A')}")
        print(f"   Word Count: {meta.get('word_count', 0):,}")
        print(f"   Processing Time: {meta.get('processing_time', 0):.1f}s ({meta.get('processing_time', 0)/60:.1f} min)")
        print(f"   Timestamp: {meta.get('processing_timestamp', 'N/A')}")
        print()
    
    # Entities
    if 'extracted_entities' in doc:
        entities = doc['extracted_entities']
        print("🔬 EXTRACTED ENTITIES:")
        print(f"   Total: {entities.get('total_entities', 0)}")
        print()
        
        for category in ['biomedical_entities', 'neuroscience_entities', 'clinical_entities', 
                        'psychology_entities', 'statistical_entities']:
            if category in entities and entities[category]:
                ents = entities[category]
                print(f"   {category.replace('_', ' ').title()}: {len(ents)}")
                # Show unique, cleaned entities
                unique_ents = list(set([e for e in ents if len(e) > 3 and not e.startswith('##')]))[:10]
                if unique_ents:
                    for ent in unique_ents:
                        print(f"      • {ent}")
                    if len(ents) > 10:
                        print(f"      ... and {len(ents)-10} more")
                print()
    
    # Rebellion Framework
    if 'rebellion_framework' in doc:
        rf = doc['rebellion_framework']
        print("🧠 REBELLION FRAMEWORK (Neurodivergence-Affirming):")
        print()
        
        sections = [
            ('narrative_deconstruction', '📖 Narrative Deconstruction'),
            ('sensory_profiling', '👁️ Sensory Profiling'),
            ('executive_function', '🎯 Executive Function'),
            ('social_communication', '💬 Social Communication'),
            ('learning_differences', '📚 Learning Differences'),
            ('emotional_regulation', '❤️ Emotional Regulation'),
            ('movement_differences', '🏃 Movement Differences'),
            ('attention_focus', '🎯 Attention & Focus'),
        ]
        
        for key, label in sections:
            if key in rf and rf[key]:
                items = rf[key]
                print(f"   {label}: ({len(items)} insights)")
                for i, item in enumerate(items[:3], 1):
                    # Truncate long items
                    item_text = item if len(item) <= 200 else item[:197] + "..."
                    print(f"      {i}. {item_text}")
                if len(items) > 3:
                    print(f"      ... and {len(items)-3} more")
                print()
    
    # Key Findings
    if 'key_findings' in doc:
        findings = doc['key_findings']
        if isinstance(findings, list) and findings:
            print("🔑 KEY FINDINGS:")
            for i, finding in enumerate(findings[:5], 1):
                finding_text = finding if len(finding) <= 200 else finding[:197] + "..."
                print(f"   {i}. {finding_text}")
            if len(findings) > 5:
                print(f"   ... and {len(findings)-5} more")
            print()
    
    # Clinical Implications
    if 'clinical_implications' in doc:
        implications = doc['clinical_implications']
        if isinstance(implications, list) and implications:
            print("🏥 CLINICAL IMPLICATIONS:")
            for i, impl in enumerate(implications[:3], 1):
                impl_text = impl if len(impl) <= 200 else impl[:197] + "..."
                print(f"   {i}. {impl_text}")
            if len(implications) > 3:
                print(f"   ... and {len(implications)-3} more")
            print()
    
    print_separator()
    print()

def main():
    kb_path = Path("/home/antons-gs/enlitens-ai/enlitens_knowledge_base.json.temp")
    
    if not kb_path.exists():
        print("❌ Knowledge base file not found!")
        print(f"   Looking for: {kb_path}")
        sys.exit(1)
    
    print("📖 Loading knowledge base...")
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    # Show summary
    view_summary(kb)
    
    documents = kb.get('documents', [])
    
    if not documents:
        print("⚠️  No documents processed yet!")
        sys.exit(0)
    
    # Interactive mode
    while True:
        print(f"\n📚 Available documents: 1-{len(documents)}")
        print("Commands:")
        print("  • Enter a number (1-{}) to view that document".format(len(documents)))
        print("  • 'all' - view all documents")
        print("  • 'summary' - show summary again")
        print("  • 'q' or 'quit' - exit")
        print()
        
        choice = input("👉 Your choice: ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
        elif choice == 'summary':
            view_summary(kb)
        elif choice == 'all':
            for i, doc in enumerate(documents, 1):
                view_document(doc, i)
                if i < len(documents):
                    input("\n⏸️  Press Enter to continue to next document...")
        elif choice.isdigit():
            doc_num = int(choice)
            if 1 <= doc_num <= len(documents):
                view_document(documents[doc_num - 1], doc_num)
            else:
                print(f"❌ Invalid document number. Choose 1-{len(documents)}")
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()

