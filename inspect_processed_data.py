#!/usr/bin/env python3
import pickle
import argparse

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def inspect_processed_data(data_path):
    """Inspect the processed data and show available demonstrations and objects."""
    print(f"Loading processed data from {data_path}...")
    processed_data = load_pickle(data_path)
    
    print(f"\nFound {len(processed_data)} demonstrations:")
    
    for demo_id, demo_data in processed_data.items():
        print(f"\nDemo ID: {demo_id}")
        
        # Left hand info
        if 'left_hand' in demo_data:
            left_shape = demo_data['left_hand']['poses_9d'].shape
            print(f"  Left hand: {left_shape} (T={left_shape[0]}, D={left_shape[1]})")
        
        # Right hand info  
        if 'right_hand' in demo_data:
            right_shape = demo_data['right_hand']['poses_9d'].shape
            print(f"  Right hand: {right_shape} (T={right_shape[0]}, D={right_shape[1]})")
        
        # Object info
        if 'objects' in demo_data:
            objects = demo_data['objects']
            print(f"  Objects ({len(objects)} total):")
            for obj_id, obj_data in objects.items():
                obj_shape = obj_data['poses_9d'].shape
                print(f"    {obj_id}: {obj_shape} (T={obj_shape[0]}, D={obj_shape[1]})")
        
        # Show only first 3 demos in detail to avoid too much output
        if list(processed_data.keys()).index(demo_id) >= 2:
            print(f"\n... and {len(processed_data) - 3} more demonstrations")
            break
    
    # Summary of first demo for training
    first_demo_id = list(processed_data.keys())[0]
    first_demo = processed_data[first_demo_id]
    first_object_id = list(first_demo['objects'].keys())[0]
    
    print(f"\n" + "="*50)
    print("RECOMMENDED FOR OVERFITTING:")
    print(f"Demo ID: {first_demo_id}")
    print(f"Object ID: {first_object_id}")
    print(f"Left hand shape: {first_demo['left_hand']['poses_9d'].shape}")
    print(f"Right hand shape: {first_demo['right_hand']['poses_9d'].shape}")
    print(f"Object shape: {first_demo['objects'][first_object_id]['poses_9d'].shape}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Inspect processed data and show available demonstrations and objects."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="data/processed_data.pkl",
        help="Path to processed data pickle file"
    )
    
    args = parser.parse_args()
    inspect_processed_data(args.data_path)

if __name__ == "__main__":
    main() 