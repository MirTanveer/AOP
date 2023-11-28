# -*- coding: utf-8 -*-
"""
Author: Mir Tanveerul Hassan
"""
#The Ensemble model is trained on the unbalanced dataset
#642 positive samples and 7268 negative samples
#80% samples for training and 20% samples for testing/independent evaluation
#The input feature vector of 70D for the meta-models and the final ensemble model is 
#the concatenated output from the 35 baseline models


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from collections import Counter

#Import 70D vector training file
X_train=pd.read_csv(r'G:/Downloads/Research/Antioxidative_Peptides/UB_Features/UB_PP_70_train.csv').iloc[:,1:].values

#Import 70D vector testing/independent file
X_test=pd.read_csv(r'G:/G:/Downloads/Research/Antioxidative_Peptides/UB_Features/UB_PP_70_ind.csv').iloc[:,1:].values

#Training dataset labels (6328 samples)
pos_labels = np.ones(514)
neg_labels = np.zeros(5814)
y_train = np.concatenate((pos_labels,neg_labels),axis=0)


#Testing dataset labels (1582 samples)
pos_labels2 = np.ones(128)
neg_labels2 = np.zeros(1454)
y_test = np.concatenate((pos_labels2,neg_labels2),axis=0)

from collections import Counter
print(Counter(y_test).items())
print(Counter(y_train).items())

#Evaluation on the test/independent dataset
def evaluate_model_test(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(X_test)
    
    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    #fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    #MCC
    mcc=matthews_corrcoef(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    total=sum(sum(cm))
    
    #accuracy=(cm[0,0]+cm[1,1])/total
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    sen= cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    # Print result
    print('\t Accuracy:', acc)
    print('\t Precision:', prec)
    print('\t Recall:', rec)
    print('\t F1 Score:', f1)
    print('\t Area Under Curve:', auc)
    print('\t Sensitivity : ',sen)
    print('\t Specificity : ', spec)
    print('\t MCC Score : ', mcc)
    print('\t Confusion Matrix:\n', cm)
    print('\n')
    print('\n')


    return

#CV evaluation on the training dataset
def evaluate_model_train(model, X_train, y_train):
    from sklearn import metrics
    conf_matrix_list_of_arrays = []
    mcc_array=[]
    #cv = KFold(n_splits=5)
    #cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    lst_accu = []
    AUC_list=[]
    
    
    prec_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='precision'))
    recall_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='recall'))
    f1_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'))
    Acc=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
    
    
    
    for train_index, test_index in cv.split(X_train, y_train): 
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index] 
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        
        model.fit(X_train_fold, y_train_fold)
        y_pred=model.predict(X_test_fold)
        
        lst_accu.append(model.score(X_test_fold, y_test_fold))
        acc=np.mean(lst_accu)
        
        conf_matrix = confusion_matrix(y_test_fold, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)
        cm = np.mean(conf_matrix_list_of_arrays, axis=0)
        
        mcc_array.append(matthews_corrcoef(y_test_fold, model.predict(X_test_fold)))
        mcc=np.mean(mcc_array, axis=0)
        
        AUC=metrics.roc_auc_score( y_test_fold, model.predict_proba(X_test_fold)[:,1])
        AUC_list.append(AUC)
        auc=np.mean(AUC_list)
        
        
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    specificity = cm[0,0]/(cm[0,1]+cm[0,0])
    sensitivity= cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    #print("\t Confusion Matrix is: \n", cm)
    print ('\t Accuracy : ', Acc)
    print('\t Sensitivity : ', sensitivity)
    print('\t Specificity : ', specificity)
    print("\t Mean of Matthews Correlation Coefficient is: ", mcc)
    print("\t The Acc value from CM is: ", acc)
    print("\t The Recall value is: ", recall_train)
    print("\t The F1 score is: ", f1_train)
    print('\t The area under curve is:',auc)
    print('\n')



cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

#Random Forest

#Using Optuna for the hyperparameter optimization 
import optuna
from sklearn.ensemble import RandomForestClassifier
def RF_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 60)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    min_samples_split= trial.suggest_int("min_samples_split", 2, 20)
    
    ## Create Model
    model = RandomForestClassifier(max_depth = max_depth, min_samples_split=min_samples_split,
                                   n_estimators = n_estimators,n_jobs=2
                                     )
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean

#Execute optuna and set hyperparameters
RF_study = optuna.create_study(direction='maximize')
RF_study.optimize(RF_objective, n_trials=200)

optimized_RF=RandomForestClassifier(**RF_study.best_params)



# ExtraTreeClassifier


from sklearn.ensemble import ExtraTreesClassifier
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 2000),
            'max_depth' : trial.suggest_int('max_depth', 10, 90),
            'max_leaf_nodes' : trial.suggest_int('max_leaf_nodes', 15, 100),
            'criterion' : trial.suggest_categorical('criterion', ['gini', 'entropy'])

    }


    # Fit the model
    etc_model = ExtraTreesClassifier(**params)
    score = cross_val_score(etc_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
etc_study = optuna.create_study(direction='maximize')
etc_study.optimize(objective, n_trials=200)

optimized_etc =ExtraTreesClassifier(**etc_study.best_params)

# XGB


from xgboost import XGBClassifier
#cv = RepeatedStratifiedKFold(n_splits=5)
import optuna
def objective(trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
        #'eval_metric': 'mlogloss',
        #'use_label_encoder': False
    }

    # Fit the model
    xgb_model = XGBClassifier(**params,  eval_metric='mlogloss')
    score = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean
#Execute optuna and set hyperparameters
XGB_study = optuna.create_study(direction='maximize')
XGB_study.optimize(objective, n_trials=200)
optimized_XGB =XGBClassifier(**XGB_study.best_params)



# LGBM

import lightgbm as lgbm
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 100), 
        'max_depth': trial.suggest_int('max_depth', 1, 100), 
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 10), 
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 
        #'objective': 'multiclass', 
        # 'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100), 
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 0
    }


    # Fit the model
    lgbm_model = lgbm.LGBMClassifier(**params)
    score = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(objective, n_trials=200)

optimized_lgbm =lgbm.LGBMClassifier(**lgbm_study.best_params)

# CatBoost


from catboost import CatBoostClassifier
def objective(trial):
    params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

#     if param["bootstrap_type"] == "Bayesian":
#         param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#     elif param["bootstrap_type"] == "Bernoulli":
#         param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    # Fit the model
    cat_model = CatBoostClassifier(**params, silent=True)
    score = cross_val_score(cat_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
cat_study = optuna.create_study(direction='maximize')
cat_study.optimize(objective, n_trials=20)

optimized_cat =CatBoostClassifier(**cat_study.best_params, silent=True)

# VotingClassifier

from sklearn.ensemble import VotingClassifier
v_clf = VotingClassifier(estimators=[('RF', optimized_RF), ('XGB', optimized_XGB), 
                                     ("Cat", optimized_cat), ('ETC', optimized_etc), 
                                     ('LGBM', optimized_lgbm)], voting='soft')

# Results 

model={'rfc': optimized_RF, 'etc':optimized_etc,
       'lgbm': optimized_lgbm, 'xgb':optimized_XGB, 
       'Cat':optimized_cat, 'Voting_Classifier':v_clf}

for key in model:
    
    print(model[key])

from termcolor import colored
for key in model:
    if key=='rfc':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)
        print(colored('===================================================', 'red'))
    elif key=='etc':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)     
        print(colored('===================================================', 'red'))
    elif key=='lgbm':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)       
        print(colored('===================================================', 'red'))
    elif key=='xgb':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)       
        print(colored('===================================================', 'red'))   
    elif key=='Cat':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)       
        print(colored('===================================================', 'red'))
    else:
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Results on the independent dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test, y_test)       
        print(colored('===================================================', 'red')) 

#Precision-Recall Curve
#5 meta-models and the ensemble model
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc
y_score1 = optimized_RF.predict_proba(X_test)[:, 1]
y_score2 = optimized_etc.predict_proba(X_test)[:, 1]
y_score3 = optimized_XGB.predict_proba(X_test)[:, 1]
y_score4 = optimized_lgbm.predict_proba(X_test)[:, 1]
y_score5 = optimized_cat.predict_proba(X_test)[:, 1]
y_score6 = v_clf.predict_proba(X_test)[:, 1]

#calculate precision and recall
precision1, recall1, thresholds1 = precision_recall_curve(y_test, y_score1)
label1 = 'RF:AUC=%.4f' % (auc(recall1, precision1))

precision2, recall2, thresholds2 = precision_recall_curve(y_test, y_score2)
label2 = 'XGB:AUC=%.4f' % (auc(recall2, precision2))

precision3, recall3, thresholds3 = precision_recall_curve(y_test, y_score3)
label3 = 'LGBM:AUC=%.4f' % (auc(recall3, precision3))

precision4, recall4, thresholds4 = precision_recall_curve(y_test, y_score4)
label4 = 'CBC:AUC=%.4f' % (auc(recall4, precision4))

precision5, recall5, thresholds5 = precision_recall_curve(y_test, y_score5)
label5 = 'ETC:AUC=%.4f' % (auc(recall5, precision5))

precision6, recall6, thresholds6 = precision_recall_curve(y_test, y_score6)
label6 = 'Ensemble.:AUC=%.4f' % (auc(recall6, precision6))


#create precision recall curve
fig, ax = plt.subplots(figsize=(8,8))
plt.plot(recall1, precision1, color='#FA8072', label=label1, lw=2)
plt.plot(recall2, precision2, color='#13EAC9', label=label2, lw=2)

plt.plot(recall3, precision3, color='#9A0EEA', label=label3, lw=2)
plt.plot(recall4, precision4, color='brown', label=label4, lw=2)

plt.plot(recall5, precision5, color='#FFA500', label=label5, lw=2)
plt.plot(recall6, precision6, color='red', label=label6, lw=2)

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.legend(loc="lower left")
#display plot
plt.grid()
