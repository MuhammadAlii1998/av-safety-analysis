from src.data_loader import load_nhtsa_data, load_nuscenes_data
from src.preprocess import preprocess_nhtsa
from src.nuscenes_features import extract_nuscenes_features
from src.modeling import train_model
from src.evaluation import evaluate_model
from src.report import generate_report

def main():
    # Load data
    nhtsa_df = load_nhtsa_data('data/NHTSA_crash_data.csv')
    nuscenes_obj = load_nuscenes_data('data/nuscenes/')

    # Preprocess data
    X, y = preprocess_nhtsa(nhtsa_df)
    nuscenes_feats = extract_nuscenes_features(nuscenes_obj)

    # Train model
    model = train_model(X, y)

    # Evaluate model
    evaluate_model(model, X, y)

    # Generate report
    generate_report(model, X, y)

if __name__ == "__main__":
    main()
