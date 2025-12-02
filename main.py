import argparse
import numpy as np
import pandas as pd
from dsfe import train_eval, config

def main():
    parser = argparse.ArgumentParser(description="DSFE Implementation")
    parser.add_argument('--subjects', type=str, default='1,2,3,4,5,6,7,8,9', help='Comma separated subject IDs')
    args = parser.parse_args()
    
    subjects = args.subjects.split(',')
    
    results = []
    
    print("Starting DSFE Analysis...")
    print(f"Config: FTA={config.USE_FTA}, RG={config.USE_RG}, FDCC={config.USE_FDCC}, "
          f"ReliefF={config.USE_RELIEFF}, Ensemble={config.USE_ENSEMBLE}")
    
    for sub in subjects:
        try:
            acc, res = train_eval.evaluate_session(sub)
            results.append({
                'Subject': sub,
                'Accuracy': acc,
                'Std': np.std(res['accuracies'])
            })
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
            import traceback
            traceback.print_exc()
            
    if results:
        df = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(df)
        print(f"Average Accuracy: {df['Accuracy'].mean():.4f}")
        
        df.to_csv('dsfe_results.csv', index=False)
        print("Results saved to dsfe_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
