import train
import predict

if __name__ == "__main__":
    print("Training Model...")
    train.main()
    
    print("\nMaking Predictions:")
    sample_news = ["Bitcoin is surging", "Inflation fears hit Wall Street", "US economy is stable"]
    predictions = predict.predict_sentiment(sample_news)
    print("Predictions:", predictions)
