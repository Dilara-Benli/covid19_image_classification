from model_utils import ModelTraining

trainer = ModelTraining(epochs=20, learning_rate=0.001)

#ann_model, history = trainer.create_model()

#trainer.save_model_and_history(ann_model, history, 'saved_models/model3.h5', 'saved_histories/history3.json') 

loaded_history = trainer.load_history('saved_histories/history3.json')

metrics = ['loss', 'accuracy']
trainer.plot_evaluation_metrics(loaded_history, metrics)

