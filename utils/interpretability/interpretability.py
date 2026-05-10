import shap
from IPython.display import display, HTML
from lime.lime_tabular import LimeTabularExplainer
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
cur_dir = Path(os.getcwd()).resolve()
proj_dir = next(parent for parent in [cur_dir, *cur_dir.parents] if (parent / "utils").exists() and (parent / "data").exists())

SOURCE_PATH = proj_dir / "doc" / "src" / "interpretability" 
SOURCE_PATH.mkdir(parents=True, exist_ok=True)

def shap_tree_explainer(X_test, model):
    """
    Based on test data, it use the SHAP to
    explain a tree model globally, showing
    which features influence positively and negatively
    the decision.
    
    Args:
        X_test (pd.DataFrame): Dataframe of the test set
        
        model (TreeModel): Tree model trained over the data
        
        
    Plot a summary and a bar plot over a sample of 100 on the test set    
    """

    if len(X_test) > 100:
        X_sample = X_test.sample(100, random_state=42)
    else:
        X_sample = X_test
    
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_sample)

    # summary plot
    shap.summary_plot(shap_val, X_sample)

    # bar plot
    shap.summary_plot(shap_val, X_sample, plot_type="bar")
    

def shap_linear_explainer(X_test, model):
    """
    Based on test data, it use the SHAP to
    explain a linear model globally, showing
    which features influence positively and negatively
    the decision.
    
    Args:
        X_test (pd.DataFrame): Dataframe of the test set
        
        model (LinearModel): Linear model trained over the data
        
        
    Plot a summary and a bar plot over the test set    
    """

    
    explainer = shap.LinearExplainer(model)
    shap_val = explainer.shap_values(X_test)

    # summary plot
    shap.summary_plot(shap_val, X_test)

    # bar plot
    shap.summary_plot(shap_val, X_test, plot_type="bar")
    

def lime_explainer(X_train, X_test, y_test, model):
    """
    Based on train and test data, it use the LIME to
    explain locally a valid and invalid reference.
    
    Args:
        X_train (pd.DataFrame): Dataframe of the train set
        
        X_test (pd.DataFrame): Dataframe of the test set
        
        y_test (pd.Series): Series containing target values of the test set
        
        model: Model trained over the data
        
        
    Plot a valid and invalid reference, explaining locally
    which features push the model to take such decisions.       
    """

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
        )


    valid_samples = X_test[y_test==1]
    invalid_samples = X_test[y_test==0]

    if not valid_samples.empty:
        #explain valid reference
        print("Valid seizure:")
        
        valid_ref = valid_samples.iloc[0].values        

        exp_val = explainer.explain_instance(
            valid_ref,
            model.predict_proba
            )

        display(HTML(exp_val.as_html()))
        
        fig = exp_val.as_pyplot_figure()
        
        plt.title("Explanation for Class 1 (Seizure)", fontsize=15)

        plt.savefig(SOURCE_PATH / 'lime_seizure.png', bbox_inches='tight', dpi=300)
        
        plt.close(fig)

    else:
        print("\nNo valid seizures samples available")

    if not invalid_samples.empty:
        #explain valid reference
        print("\nInvalid seizure")

        invalid_ref = invalid_samples.iloc[0].values        

        exp_inval = explainer.explain_instance(
            invalid_ref,
            model.predict_proba
            )

        display(HTML(exp_inval.as_html()))
        
        fig = exp_inval.as_pyplot_figure()
        
        plt.title("Explanation for Class 0 (No Seizure)", fontsize=15)

        plt.savefig(SOURCE_PATH / 'lime_non_seizure.png', bbox_inches='tight', dpi=300)
        
        plt.close(fig)
        
    else:
        print("\nNo invalid seizures samples available")
        

def shap_deep_learning(model, X_train, X_test):
    # 1. Converti i dati in tensori PyTorch (se non lo sono già)
    # Prendiamo un piccolo campione per il background (es. 100 righe)
    background = torch.tensor(X_train.iloc[:100].values).float()
    test_samples = torch.tensor(X_test.iloc[:10].values).float()

    # 2. Inizializza l'explainer specifico per il Deep Learning
    explainer = shap.DeepExplainer(model, background)

    # 3. Calcola i valori SHAP
    shap_values = explainer.shap_values(test_samples)

    # 4. Visualizzazione
    # Se i dati sono piatti (come le feature che abbiamo visto prima)
    shap.summary_plot(shap_values, test_samples, feature_names=X_test.columns)
    

def explain_model(X_train, X_test, y_test, model):

    model_name = type(model).__name__
    
    print(f"\n|--- Explainability of the model {model_name} ---|")

    # 1. TreeBased models
    tree_models = ['RandomForestModel', 'XGBModel', 'LGBModel']
    
    # 2. LinearBased models
    linear_models = ['LogisticRegressionModel', 'RidgeClassifier']
    
    # 3. Deep Learning models
    dl_models = ['ConvNetModel', 'RecurrentNetModel']

    # 4. Instance models
    inst_models = ['KNNModel', 'SVMModel', 'NaiveBayesModel']

    internal_model = model.model    

    if model_name in tree_models:
        print("\nExplain treebased model")
        shap_tree_explainer(X_test=X_test, model=internal_model)
        
    elif model_name in linear_models:
        print("\nExplain linearbased model")
        shap_linear_explainer(X_test=X_test, model=internal_model)
        
    elif model_name in dl_models:
        print("\nExplain deeplearning model")
        shap_deep_learning(internal_model, X_train, X_test)
        
    elif model_name in inst_models:
        print("\nExplain instance model")
        lime_explainer(X_train=X_train, X_test=X_test, y_test=y_test, model=internal_model)