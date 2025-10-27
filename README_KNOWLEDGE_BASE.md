# Enlitens AI Knowledge Base System

This system processes neuroscience research papers and extracts EVERYTHING possible for marketing, SEO, website copy, blog content, social media, educational content, clinical applications, research, and content creation.

## 🚀 Quick Start

### Initial Processing (All PDFs)
```bash
# Process all PDFs in the input directory
python process_knowledge_base.py --input-dir enlitens_corpus/input_pdfs --output-file knowledge_base.json

# Process with exports for different use cases
python process_knowledge_base.py --input-dir enlitens_corpus/input_pdfs --output-file knowledge_base.json --export
```

### Adding New PDFs (Incremental Updates)
```bash
# Add new PDFs to existing knowledge base
python add_new_pdfs.py --input-dir enlitens_corpus/input_pdfs --knowledge-base knowledge_base.json
```

## 📊 What Gets Extracted

### Marketing Content
- **Headlines**: Attention-grabbing titles for campaigns
- **Taglines**: Short, memorable phrases
- **Value Propositions**: Key benefits and differentiators
- **Benefits**: Specific advantages for clients
- **Pain Points**: Problems the research addresses
- **Social Proof**: Research-backed credibility
- **Call to Actions**: Compelling next steps
- **Target Audience**: Who this applies to
- **Unique Selling Points**: What makes this special

### SEO Content
- **Primary Keywords**: Main search terms
- **Secondary Keywords**: Related terms
- **Long Tail Keywords**: Specific phrases
- **Meta Descriptions**: Search result descriptions
- **Title Tags**: Page titles
- **Heading Tags**: Section headers
- **Internal Linking**: Content connections
- **External Linking**: Reference opportunities
- **Content Topics**: Subject areas

### Website Copy
- **About Sections**: Company/service descriptions
- **Feature Descriptions**: What you offer
- **Benefit Statements**: Why it matters
- **Testimonials**: Client feedback
- **FAQ Content**: Common questions
- **Service Descriptions**: What you do
- **Pricing Justification**: Value explanations
- **Contact Information**: How to reach you

### Blog Content
- **Article Ideas**: Blog post topics
- **Blog Outlines**: Post structures
- **Talking Points**: Key discussion points
- **Expert Quotes**: Authoritative statements
- **Statistics**: Data points
- **Case Studies**: Real examples
- **How-to Guides**: Step-by-step instructions
- **Myth Busting**: Correcting misconceptions

### Social Media Content
- **Post Ideas**: Social media topics
- **Captions**: Image descriptions
- **Quotes**: Shareable content
- **Hashtags**: Search tags
- **Story Ideas**: Narrative content
- **Reel Ideas**: Video content
- **Carousel Content**: Multi-slide posts
- **Poll Questions**: Engagement content

### Educational Content
- **Explanations**: Clear descriptions
- **Examples**: Real-world applications
- **Analogies**: Comparisons
- **Definitions**: Term explanations
- **Processes**: Step-by-step procedures
- **Comparisons**: Side-by-side analysis
- **Visual Aids**: Diagram descriptions
- **Learning Objectives**: Educational goals

### Clinical Content
- **Interventions**: Therapeutic approaches
- **Assessments**: Evaluation methods
- **Outcomes**: Expected results
- **Protocols**: Treatment procedures
- **Guidelines**: Best practices
- **Contraindications**: Safety warnings
- **Side Effects**: Potential issues
- **Monitoring**: Progress tracking

### Research Content
- **Findings**: Key discoveries
- **Statistics**: Data points
- **Methodologies**: Research methods
- **Limitations**: Study constraints
- **Future Directions**: Next steps
- **Implications**: What it means
- **Citations**: References
- **References**: Source materials

### Content Creation
- **Topic Ideas**: Content themes
- **Angle Ideas**: Different perspectives
- **Hook Ideas**: Attention grabbers
- **Story Ideas**: Narrative content
- **Series Ideas**: Multi-part content
- **Collaboration Ideas**: Partnership opportunities
- **Trend Ideas**: Current topics
- **Seasonal Ideas**: Time-based content

## 🔧 System Components

### Core Files
- `src/extraction/comprehensive_extractor.py` - Main extraction engine
- `src/knowledge_base/knowledge_manager.py` - Database management
- `process_knowledge_base.py` - Main processing script
- `add_new_pdfs.py` - Incremental update script

### Test Files
- `test_comprehensive_extraction.py` - Test extraction
- `demo_translation_process.py` - Show translation process

## 📈 Usage Examples

### Process All PDFs
```bash
# Process all PDFs and create knowledge base
python process_knowledge_base.py --input-dir enlitens_corpus/input_pdfs --output-file knowledge_base.json --export
```

### Add New PDFs
```bash
# Add new PDFs to existing knowledge base
python add_new_pdfs.py --input-dir enlitens_corpus/input_pdfs --knowledge-base knowledge_base.json
```

### Search Knowledge Base
```bash
# Search for specific content
python process_knowledge_base.py --search "anxiety treatment" --knowledge-base knowledge_base.json
```

### Export for Specific Use Cases
```bash
# Export marketing content
python process_knowledge_base.py --export --export-dir ./exports --knowledge-base knowledge_base.json
```

## 📁 Output Files

### Main Knowledge Base
- `knowledge_base.json` - Complete knowledge base
- `processed_files.json` - Registry of processed files

### Exports (if --export flag used)
- `exports/knowledge_base_marketing.json` - Marketing content
- `exports/knowledge_base_content_creation.json` - Content creation
- `exports/knowledge_base_educational.json` - Educational content
- `exports/knowledge_base_clinical.json` - Clinical content
- `exports/knowledge_base_complete.json` - All content

## 🔄 Future-Proofing

The system is designed to grow with your needs:

1. **Drop new PDFs** into the input directory
2. **Run the add script** to process only new files
3. **Knowledge base grows** automatically
4. **No reprocessing** of existing files
5. **Quality maintained** through validation

## 📊 Quality Metrics

Each extraction includes quality scores:
- **Completeness Score**: How much content was extracted
- **Structure Score**: How well organized the content is
- **Metadata Score**: How complete the source information is
- **Content Score**: How rich the content is

## 🎯 Use Cases

### Marketing Teams
- Headlines for campaigns
- Value propositions for sales
- Social proof for credibility
- Target audience insights

### Content Creators
- Blog post ideas
- Social media content
- Educational materials
- Story ideas

### SEO Specialists
- Keyword research
- Meta descriptions
- Content topics
- Internal linking opportunities

### Clinical Teams
- Intervention strategies
- Assessment tools
- Outcome measures
- Safety considerations

### Research Teams
- Methodology insights
- Statistical findings
- Future directions
- Collaboration opportunities

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Place PDFs** in `enlitens_corpus/input_pdfs/`
3. **Run initial processing**: `python process_knowledge_base.py --input-dir enlitens_corpus/input_pdfs --output-file knowledge_base.json --export`
4. **Add new PDFs** as needed: `python add_new_pdfs.py --input-dir enlitens_corpus/input_pdfs --knowledge-base knowledge_base.json`

## 📞 Support

The system is designed to be self-maintaining and scalable. It will:
- Process new PDFs automatically
- Maintain quality standards
- Grow your knowledge base
- Provide rich content for all use cases

Your knowledge base will become a comprehensive resource for all your content and marketing needs!
