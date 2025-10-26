def generate_report(model, X, y):
    import matplotlib.pyplot as plt
    model.plot_importance(max_num_features=10)
    plt.title('Feature Importance')
    plt.show()
