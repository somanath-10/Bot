import json
from pathlib import Path
from bot import compose

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_submission():
    expanded_dir = Path("dataset/expanded")
    test_pairs_path = expanded_dir / "test_pairs.json"
    
    if not test_pairs_path.exists():
        print(f"Error: {test_pairs_path} not found. Did you run generate_dataset.py?")
        return

    test_pairs = load_json(test_pairs_path).get("pairs", [])
    
    if not test_pairs:
        print("Error: No test pairs found.")
        return

    print(f"Loaded {len(test_pairs)} test pairs. Generating messages...")

    # Preload all contexts into memory
    categories = {}
    for p in (expanded_dir / "categories").glob("*.json"):
        cat_data = load_json(p)
        categories[cat_data.get("slug")] = cat_data

    merchants = {}
    for p in (expanded_dir / "merchants").glob("*.json"):
        m_data = load_json(p)
        merchants[m_data.get("merchant_id")] = m_data
        
    customers = {}
    for p in (expanded_dir / "customers").glob("*.json"):
        c_data = load_json(p)
        customers[c_data.get("customer_id")] = c_data
        
    triggers = {}
    for p in (expanded_dir / "triggers").glob("*.json"):
        t_data = load_json(p)
        triggers[t_data.get("id")] = t_data

    submissions = []
    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")
        
        trigger = triggers.get(trigger_id)
        merchant = merchants.get(merchant_id)
        category = categories.get(merchant.get("category_slug"))
        customer = customers.get(customer_id) if customer_id else None
        
        print(f"Generating for {test_id} (trigger: {trigger_id})...")

        res_json = compose(category, merchant, trigger, customer)
        body = res_json.get("body", "Failed to generate")
        cta = res_json.get("cta", "open_ended")
        send_as = res_json.get("send_as", "vera")
        rationale = res_json.get("rationale", "Generated successfully")

        submissions.append({
            "test_id": test_id,
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": rationale
        })

    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for s in submissions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(submissions)} submissions to submission.jsonl")

if __name__ == "__main__":
    generate_submission()
