# -*- coding: utf-8 -*-
"""main2.py

Main script to extract data and dump to recommendation_engine database
"""

from production_data_extractor_fixed import KeekDataExtractor

def main():
    """Main function to run complete data pipeline"""
    print("🚀 STARTING COMPLETE DATA PIPELINE")
    print("="*60)
    print("📊 Steps:")
    print("   1. Extract data from production database")
    print("   2. Save RAW data to CSV files")
    print("   3. Dump ENHANCED data to recommendation_engine database")
    print("   4. Save ENHANCED data to CSV files")
    print("="*60)
    
    # Initialize the extractor
    extractor = KeekDataExtractor()
    
    # Step 1: Extract data from production database
    print("\n📥 STEP 1: Extracting data from production database...")
    interaction_df, user_df, post_df = extractor.extract_complete_dataset(sample_size=None)
    
    if interaction_df is not None:
        # Step 2: Save RAW data to CSV files
        print("\n💾 STEP 2: Saving RAW data to CSV files...")
        interaction_file, user_file, post_file = extractor.save_data_to_csv(
            interaction_df, user_df, post_df, 
            output_dir='production_data'
        )
        
        # Step 3: Dump ENHANCED data to recommendation_engine database
        print("\n🗄️ STEP 3: Dumping ENHANCED data to recommendation_engine database...")
        enhanced_interaction_df, enhanced_user_df = extractor.dump_data_to_recommendation_db(interaction_df, user_df, post_df)
        
        if enhanced_interaction_df is not None:
            # Step 4: Save ENHANCED data to CSV files
            print("\n💾 STEP 4: Saving ENHANCED data to CSV files...")
            enhanced_interaction_file, enhanced_user_file, enhanced_post_file = extractor.save_enhanced_data_to_csv(
                enhanced_interaction_df, enhanced_user_df, post_df,
                output_dir='enhanced_data'
            )
            
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("📊 FINAL SUMMARY:")
            print(f"   📥 Extracted from production:")
            print(f"      - Interactions: {len(interaction_df):,}")
            print(f"      - Users: {len(user_df):,}")
            print(f"      - Posts: {len(post_df):,}")
            print(f"   💾 RAW CSV files saved in: production_data/")
            print(f"   🗄️ ENHANCED data dumped to: recommendation_engine database")
            print(f"   💾 ENHANCED CSV files saved in: enhanced_data/")
            print(f"   🔤 Language features: {len(extractor.lang_encoder.classes_)} unique languages encoded")
            print("="*60)
        else:
            print("❌ Failed to dump data to recommendation_engine database!")
    else:
        print("❌ Data extraction failed!")

if __name__ == "__main__":
    main()