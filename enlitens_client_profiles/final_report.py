"""
Generate final report on all 100 generated personas.
Shows diversity, quality metrics, and summary.
"""
import json
from pathlib import Path
from collections import Counter

def main():
    project_root = Path("/home/antons-gs/enlitens-ai")
    profiles_dir = project_root / "enlitens_client_profiles" / "profiles"
    
    # Load all cluster personas
    persona_files = sorted(profiles_dir.glob("persona_cluster_*.json"))
    
    if not persona_files:
        print("❌ No personas found!")
        return
    
    print("="*80)
    print(f"FINAL REPORT: {len(persona_files)} PERSONAS GENERATED")
    print("="*80 + "\n")
    
    personas = []
    for f in persona_files:
        with open(f) as fp:
            personas.append(json.load(fp))
    
    # Analyze diversity
    ages = []
    genders = []
    situations = []
    localities = []
    identities = []
    
    for p in personas:
        demo = p.get("identity_demographics", {})
        neuro = p.get("neurodivergence_mental_health", {})
        
        ages.append(demo.get("age_range", "Unknown"))
        genders.append(demo.get("gender", "Unknown"))
        situations.append(demo.get("current_life_situation", "Unknown")[:50])
        localities.append(demo.get("locality", "Unknown"))
        identities.extend(neuro.get("identities", []))
    
    print("DIVERSITY ANALYSIS")
    print("-"*80 + "\n")
    
    print("Age Distribution:")
    for age, count in Counter(ages).most_common():
        pct = (count / len(personas)) * 100
        print(f"  {age:20} {count:3} ({pct:5.1f}%)")
    
    print("\nGender Distribution:")
    for gender, count in Counter(genders).most_common():
        pct = (count / len(personas)) * 100
        print(f"  {gender:20} {count:3} ({pct:5.1f}%)")
    
    print("\nTop 10 Life Situations:")
    for situation, count in Counter(situations).most_common(10):
        print(f"  • {situation}")
    
    print("\nLocality Distribution:")
    for locality, count in Counter(localities).most_common():
        pct = (count / len(personas)) * 100
        print(f"  {locality:30} {count:3} ({pct:5.1f}%)")
    
    print("\nNeurodivergent Identities:")
    for identity, count in Counter(identities).most_common():
        print(f"  {identity:30} {count:3}")
    
    # Sample personas
    print("\n" + "="*80)
    print("SAMPLE PERSONAS (First 20)")
    print("="*80 + "\n")
    
    for i, p in enumerate(personas[:20], 1):
        meta = p.get("meta", {})
        demo = p.get("identity_demographics", {})
        print(f"{i:2}. {meta.get('persona_name')}")
        print(f"    {demo.get('age_range')} | {demo.get('gender')} | {demo.get('current_life_situation')}")
        print(f"    {demo.get('locality')}")
        print()
    
    # Quality check
    print("="*80)
    print("QUALITY METRICS")
    print("="*80 + "\n")
    
    total_chars = 0
    total_fields = 0
    has_dev_story = 0
    has_food_details = 0
    
    for p in personas:
        # Count characters
        total_chars += len(json.dumps(p))
        
        # Count filled fields
        dev = p.get("developmental_story", {})
        if dev.get("childhood_environment"):
            has_dev_story += 1
        
        ef = p.get("executive_function_sensory", {})
        if ef.get("food_sensory_details") and len(ef.get("food_sensory_details", "")) > 50:
            has_food_details += 1
    
    avg_chars = total_chars / len(personas)
    
    print(f"Average persona size: {avg_chars:,.0f} characters")
    print(f"Personas with developmental story: {has_dev_story}/{len(personas)} ({has_dev_story/len(personas)*100:.1f}%)")
    print(f"Personas with detailed food sensory: {has_food_details}/{len(personas)} ({has_food_details/len(personas)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE")
    print("="*80)
    print(f"\nSuccessfully generated {len(personas)} unique, deeply human client personas.")
    print(f"Each persona represents a real client segment from your intake data.")
    print(f"\nFiles saved to: {profiles_dir}")
    print()


if __name__ == "__main__":
    main()

