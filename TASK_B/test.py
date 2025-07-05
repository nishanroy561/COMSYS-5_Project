import os
import torch
from collections import Counter

def print_header(title):
    print("=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)

def print_section(title):
    print(f"\n📊 {title}")
    print("-" * 40)

def analyze_embeddings():
    """Analyze the training embeddings file"""
    print_header("DATASET ANALYSIS REPORT")
    
    # Load train embeddings
    try:
        emb = torch.load("train_embeddings.pt")
        print(f"✅ Successfully loaded train_embeddings.pt")
    except Exception as e:
        print(f"❌ Error loading train_embeddings.pt: {e}")
        return
    
    print_section("TRAINING DATA STATISTICS")
    print(f"👥 Total people in training: {len(emb):,}")
    
    # Analyze embedding counts per person
    embedding_counts = [len(embeddings) for embeddings in emb.values()]
    print(f"📸 Total face embeddings: {sum(embedding_counts):,}")
    print(f"📊 Average embeddings per person: {sum(embedding_counts)/len(embedding_counts):.1f}")
    print(f"📈 Max embeddings per person: {max(embedding_counts)}")
    print(f"📉 Min embeddings per person: {min(embedding_counts)}")
    
    # Show sample names with counts
    print(f"\n📋 Sample people (with embedding counts):")
    sample_items = list(emb.items())[:10]
    for name, embeddings in sample_items:
        print(f"   • {name}: {len(embeddings)} embeddings")

def analyze_validation_data():
    """Analyze the validation dataset"""
    print_section("VALIDATION DATA STATISTICS")
    
    val_dir = "val"
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory '{val_dir}' not found!")
        return
    
    val_people = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    print(f"🧪 Total people in validation: {len(val_people):,}")
    
    # Count images in validation
    total_val_images = 0
    person_image_counts = {}
    
    for person in val_people:
        person_path = os.path.join(val_dir, person)
        images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        
        # Check for distortion folder
        distort_path = os.path.join(person_path, 'distortion')
        if os.path.exists(distort_path):
            distort_images = [f for f in os.listdir(distort_path) if f.endswith('.jpg')]
            images.extend([f'distortion/{img}' for img in distort_images])
        
        person_image_counts[person] = len(images)
        total_val_images += len(images)
    
    print(f"📸 Total validation images: {total_val_images:,}")
    print(f"📊 Average images per person: {total_val_images/len(val_people):.1f}")
    
    # Show sample validation people
    print(f"\n📋 Sample validation people (with image counts):")
    sample_val = list(person_image_counts.items())[:10]
    for name, count in sample_val:
        print(f"   • {name}: {count} images")
    
    return val_people, person_image_counts

def analyze_data_overlap():
    """Analyze overlap between training and validation data"""
    print_section("DATA OVERLAP ANALYSIS")
    
    # Load embeddings
    emb = torch.load("train_embeddings.pt")
    train_people = set(emb.keys())
    
    # Get validation people
    val_dir = "val"
    val_people = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    val_people_set = set(val_people)
    
    # Find overlaps and differences
    common_people = train_people.intersection(val_people_set)
    train_only = train_people - val_people_set
    val_only = val_people_set - train_people
    
    print(f"🔄 People in both train and val: {len(common_people):,}")
    print(f"📚 People only in training: {len(train_only):,}")
    print(f"🧪 People only in validation: {len(val_only):,}")
    
    # Calculate coverage percentages
    train_coverage = len(common_people) / len(val_people) * 100
    val_coverage = len(common_people) / len(train_people) * 100
    
    print(f"📊 Validation coverage: {train_coverage:.1f}% (people in val that are also in train)")
    print(f"📊 Training coverage: {val_coverage:.1f}% (people in train that are also in val)")
    
    # Show examples of missing people
    if val_only:
        print(f"\n❌ People in validation but missing from training:")
        for name in list(val_only)[:10]:
            print(f"   • {name}")
        if len(val_only) > 10:
            print(f"   ... and {len(val_only) - 10} more")
    
    if train_only:
        print(f"\n📚 People in training but not in validation:")
        for name in list(train_only)[:10]:
            print(f"   • {name}")
        if len(train_only) > 10:
            print(f"   ... and {len(train_only) - 10} more")

def analyze_file_structure():
    """Analyze the overall file structure"""
    print_section("FILE STRUCTURE ANALYSIS")
    
    # Check main directories
    directories = ['train', 'val']
    for dir_name in directories:
        if os.path.exists(dir_name):
            person_count = len([d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))])
            print(f"📁 {dir_name}/: {person_count:,} person folders")
        else:
            print(f"❌ {dir_name}/: Directory not found")
    
    # Check embedding file
    if os.path.exists("train_embeddings.pt"):
        file_size = os.path.getsize("train_embeddings.pt") / (1024 * 1024)  # MB
        print(f"💾 train_embeddings.pt: {file_size:.1f} MB")
    else:
        print(f"❌ train_embeddings.pt: File not found")

def main():
    """Main analysis function"""
    try:
        analyze_file_structure()
        analyze_embeddings()
        analyze_validation_data()
        analyze_data_overlap()
        
        print("\n" + "=" * 60)
        print("✅ DATASET ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
