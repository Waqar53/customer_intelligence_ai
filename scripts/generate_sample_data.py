#!/usr/bin/env python3
"""
Sample Data Generator for Customer Intelligence AI.

This script creates realistic sample customer feedback data
for testing and demonstration purposes.

Run this script to populate the data/sample/ directory:
    python scripts/generate_sample_data.py
"""

import os
import csv
import random
from datetime import datetime, timedelta

# Sample feedback templates organized by category
FEEDBACK_TEMPLATES = {
    "slow_performance": [
        "The app is extremely slow when loading my dashboard. Takes over 30 seconds!",
        "Page load times are unacceptable. I've been waiting minutes for simple actions.",
        "Performance has degraded significantly since the last update. Very frustrating.",
        "The search feature is painfully slow. I can't find anything quickly anymore.",
        "Loading data takes forever. This is killing my productivity.",
        "App freezes constantly when I try to export reports.",
        "The mobile app is so slow it's basically unusable.",
        "Response times have gotten worse over the past month.",
    ],
    "poor_support": [
        "Customer support never responds to my tickets. Been waiting 2 weeks!",
        "I can't get anyone to help me with my billing issue.",
        "Support chat always says 'all agents busy' - what's the point?",
        "Had to explain my issue to 5 different agents. Still not resolved.",
        "Your support team doesn't seem to understand the product.",
        "Been transferred 4 times and still haven't solved my problem.",
        "Email support takes days to respond. Unacceptable for paid plan.",
        "Support documentation is outdated and unhelpful.",
    ],
    "missing_features": [
        "We really need bulk edit functionality. Processing items one by one is tedious.",
        "Why is there no dark mode? My eyes hurt using this at night.",
        "Please add export to PDF. Excel is not enough for our needs.",
        "Need API access on the basic plan. Can't afford enterprise.",
        "Would love to see integration with Slack for notifications.",
        "Calendar view is missing. Hard to plan without it.",
        "Custom fields would make this product so much better.",
        "Mobile app is missing half the features of the web version.",
    ],
    "bugs": [
        "Found a bug: clicking save sometimes loses all my changes.",
        "The export function creates corrupted files half the time.",
        "Notifications aren't working at all on iOS.",
        "Login fails randomly even with correct credentials.",
        "Data sync between devices is broken. Seeing old information.",
        "Charts display wrong numbers. Totals don't add up.",
        "Can't upload files larger than 1MB despite 10MB limit shown.",
        "Dropdown menus don't work on Safari browser.",
    ],
    "pricing": [
        "Pricing is too high for small businesses like ours.",
        "Why did you increase prices by 40% with no new features?",
        "The free tier is too limited to even evaluate the product properly.",
        "Hidden fees for basic features is very disappointing.",
        "Competitors offer the same for half the price.",
        "Monthly billing should be available, not just annual.",
        "Enterprise pricing is completely out of our budget.",
        "Feel like I'm paying for features I don't use.",
    ],
    "positive": [
        "Love this product! Has transformed how our team works.",
        "Best investment we made this year. ROI has been amazing.",
        "Support team resolved my issue in under an hour. Impressed!",
        "The new dashboard is beautiful and so easy to use.",
        "Finally a product that just works. No complaints.",
        "Been using this for 2 years and it keeps getting better.",
        "Recommended this to all my colleagues. They love it too.",
        "The onboarding experience was smooth and helpful.",
    ],
}

# Customer name templates
FIRST_NAMES = ["John", "Sarah", "Mike", "Emily", "David", "Lisa", "James", "Anna", "Robert", "Maria"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Taylor"]

# Plan types
PLANS = ["free", "basic", "professional", "enterprise"]


def generate_sample_csv(output_path: str, num_records: int = 100):
    """Generate a CSV file with sample customer feedback."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    records = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(num_records):
        # Pick a random category and template
        category = random.choice(list(FEEDBACK_TEMPLATES.keys()))
        feedback = random.choice(FEEDBACK_TEMPLATES[category])
        
        # Generate random customer info
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        customer_name = f"{first_name} {last_name}"
        
        # Random date within last 90 days
        days_ago = random.randint(0, 90)
        feedback_date = (start_date + timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Random plan and rating
        plan = random.choice(PLANS)
        rating = 5 if category == "positive" else random.randint(1, 3)
        
        records.append({
            "id": i + 1,
            "customer_name": customer_name,
            "feedback": feedback,
            "category": category,
            "date": feedback_date,
            "plan": plan,
            "rating": rating,
        })
    
    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Generated {num_records} feedback records in {output_path}")
    return output_path


def generate_sample_text(output_path: str):
    """Generate a text file with sample feedback."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    feedback_list = []
    for category in ["slow_performance", "poor_support", "bugs"]:
        feedback_list.extend(random.sample(FEEDBACK_TEMPLATES[category], 3))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Customer Feedback Collection\n")
        f.write("=" * 40 + "\n\n")
        for i, feedback in enumerate(feedback_list, 1):
            f.write(f"{i}. {feedback}\n\n")
    
    print(f"Generated text feedback file: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate sample data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    sample_dir = os.path.join(project_root, "data", "sample")
    
    # Create main CSV with 100 records
    generate_sample_csv(
        os.path.join(sample_dir, "feedback.csv"),
        num_records=100
    )
    
    # Create smaller CSV for quick testing
    generate_sample_csv(
        os.path.join(sample_dir, "feedback_small.csv"),
        num_records=20
    )
    
    # Create text file sample
    generate_sample_text(
        os.path.join(sample_dir, "feedback_notes.txt")
    )
    
    print("\nSample data generation complete!")
    print(f"Files created in: {sample_dir}")
