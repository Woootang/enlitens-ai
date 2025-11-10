# NER Models Implemented - November 9, 2025

## **5-Model Ensemble for Neurodivergence Research**

### **Models Deployed:**

1. **OpenMed DiseaseDetect-SuperClinical-434M**
   - **Purpose:** Diseases & conditions (autism, ADHD, anxiety, alexithymia)
   - **Size:** 434M params (~0.85GB VRAM)
   - **Accuracy:** 91.2% F1 on BC5CDR-disease
   - **License:** Apache 2.0
   - **HF URL:** `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M`

2. **OpenMed PharmaDetect-SuperClinical-434M**
   - **Purpose:** Chemicals, drugs, neurotransmitters (dopamine, serotonin, medications)
   - **Size:** 434M params (~0.85GB VRAM)
   - **Accuracy:** 96.1% F1 on BC5CDR-chem
   - **License:** Apache 2.0
   - **HF URL:** `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M`

3. **OpenMed AnatomyDetect-ElectraMed-560M**
   - **Purpose:** Brain regions & anatomical structures (amygdala, prefrontal cortex, hippocampus)
   - **Size:** 560M params (~1.1GB VRAM)
   - **Accuracy:** 90.6% F1 on AnatEM
   - **License:** Apache 2.0
   - **HF URL:** `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-560M`

4. **OpenMed GenomeDetect-SuperClinical-434M**
   - **Purpose:** Genes & proteins (SHANK3, OXTR, MECP2)
   - **Size:** 434M params (~0.85GB VRAM)
   - **Accuracy:** 90.1% F1 on BC2GM
   - **License:** Apache 2.0
   - **HF URL:** `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M`

5. **ClinicalDistilBERT i2b2-2010**
   - **Purpose:** Clinical symptoms (PROBLEM), treatments (TREATMENT), tests (TEST)
   - **Size:** 65M params (~0.13GB VRAM)
   - **Accuracy:** 89% F1 on i2b2-2010
   - **License:** MIT
   - **HF URL:** `nlpie/clinical-distilbert-i2b2-2010`

---

## **Architecture:**

### **Sequential Loading Strategy:**
```
vLLM (Qwen 2.5-14B): 22GB VRAM (stays loaded)
↓
For each PDF:
  1. Load DiseaseDetect → Extract → Unload → Clear cache
  2. Load PharmaDetect → Extract → Unload → Clear cache
  3. Load AnatomyDetect → Extract → Unload → Clear cache
  4. Load GenomeDetect → Extract → Unload → Clear cache
  5. Load ClinicalDistilBERT → Extract → Unload → Clear cache
```

### **Memory Budget:**
- **Peak VRAM:** ~23.1GB (22GB Qwen + 1.1GB AnatomyDetect)
- **Safe on:** RTX 3090 24GB
- **Per-model limit:** <2GB

---

## **Entity Coverage:**

| Category | Example Entities | Model |
|----------|-----------------|-------|
| **Conditions** | autism, ADHD, anxiety, depression, alexithymia | DiseaseDetect |
| **Chemicals** | dopamine, serotonin, oxytocin, medications | PharmaDetect |
| **Anatomy** | amygdala, prefrontal cortex, hippocampus, insula | AnatomyDetect |
| **Genes** | SHANK3, OXTR, MECP2, SLC6A4 | GenomeDetect |
| **Symptoms** | executive dysfunction, sensory overload, meltdowns | ClinicalDistilBERT |
| **Treatments** | CBT, occupational therapy, medication names | ClinicalDistilBERT |
| **Tests** | ADOS, fMRI, WISC-IV, questionnaires | ClinicalDistilBERT |

---

## **Performance Expectations:**

### **Speed:**
- **Per PDF:** ~30 seconds added (5 models × ~6 seconds each)
- **Total (345 PDFs):** ~3 hours additional processing time
- **Total pipeline:** ~6 hours (vs 3 hours without NER)

### **Quality:**
- **Precision:** 90-96% (per benchmark)
- **Recall:** High (ensemble covers gaps)
- **False Positives:** Low (domain-specific training)

---

## **Implementation Details:**

### **GPU Memory Management:**
- Dynamic loading/unloading via `GPUMemoryManager`
- `torch.cuda.empty_cache()` after each model
- Memory monitoring with `log_memory_stats()`
- Automatic cleanup on low memory

### **Error Handling:**
- Try/except on each model
- Fallback to CPU if OOM
- Graceful degradation (missing entities logged, not fatal)
- Batch size reduction on OOM

### **Optimization:**
- FP16 precision (half memory, minimal accuracy loss)
- Text truncation to 2000 chars (speed optimization)
- `aggregation_strategy="simple"` (merge sub-tokens)
- Sequential execution (no concurrent GPU contention)

---

## **Testing Strategy:**

### **Validation Papers:**
1. **Autism neuroscience** (brain regions, fMRI)
2. **ADHD dopamine** (neurotransmitters, receptors)
3. **Autism psychology** (alexithymia, sensory traits)
4. **Multi-condition** (autism + anxiety + ADHD)

### **Success Criteria:**
- ✅ Diseases extracted (autism, ADHD, anxiety)
- ✅ Brain regions extracted (amygdala, prefrontal cortex)
- ✅ Neurotransmitters extracted (dopamine, serotonin)
- ✅ Genes extracted (if present)
- ✅ Symptoms extracted (executive dysfunction, sensory issues)
- ✅ Treatments extracted (CBT, medications)

---

## **Known Limitations:**

1. **Acronyms:** May miss test acronyms (ADOS, AQ, SRS) - fallback dictionary planned
2. **Alexithymia:** May not be tagged as disease (considered trait) - covered by clinical model as PROBLEM
3. **Colloquial terms:** Some informal symptom descriptions may be missed
4. **Context window:** Limited to 2000 chars per model (optimization trade-off)

---

## **Next Steps (Future Enhancements):**

1. **GLiNER-BioMed** (optional 6th model for custom entity types)
   - Zero-shot NER for "psychological concepts", "neuropsychological tests"
   - 86M params (~0.97GB VRAM)
   - Use for entities not covered by other models

2. **Dictionary fallback** for known test acronyms (ADOS, AQ, SRS, BRIEF, etc.)

3. **Entity linking** to UMLS/ontologies (post-processing)

4. **Confidence scoring** integration with existing system

---

## **References:**

- OpenMed NER Suite: https://huggingface.co/OpenMed
- ClinicalDistilBERT: https://huggingface.co/nlpie/clinical-distilbert-i2b2-2010
- Research source: ChatGPT Pro Deep Research (November 2025)

---

**Status:** ✅ Implemented (November 9, 2025)  
**Deployed in:** `src/agents/extraction_team.py`  
**GPU Manager:** `src/utils/gpu_memory_manager.py`

