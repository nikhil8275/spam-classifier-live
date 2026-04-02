# Spam Classifier Neural Network Update Plan

## Completed:
- [x] Implemented Neural Network (MLPClassifier) in train_model.py
- [x] Updated TODO.md with progress tracking

## Next Steps:
1. [ ] Run `python train_model.py` to train NN model and generate new model.pkl/vectorizer.pkl (activates NLTK downloads if needed)
2. [ ] Run `python test_prediction.py` to verify predictions
3. [ ] Test web app: `cd spam_web && ..\\spam_env\\Scripts\\activate.bat && python app.py` then visit http://127.0.0.1:5000
   - the app now loads `model.pkl` and `vectorizer.pkl` from the project root automatically
4. [ ] Compare NN accuracy vs original NB (expect ~96-98% accuracy)

**Note:** Neural Network replaces MultinomialNB for potentially better performance. Preprocessing and TF-IDF unchanged for compatibility.
