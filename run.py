from model_utils import ModelTraining

trainer = ModelTraining(epochs=20, learning_rate=0.001)

# ann_model, history = trainer.create_model('saved_models/model2.h5')

# trainer.save_history(history, 'saved_histories/history2.json')

loaded_model = trainer.load_model('saved_models/model2.h5')
loaded_history = trainer.load_history('saved_histories/history2.json')

metrics = ['loss', 'accuracy']
trainer.plot_evaluation_metrics(loaded_history, metrics)
trainer.visualize_predictions(loaded_model)
trainer.plot_confusion_matrix_and_report(loaded_model)