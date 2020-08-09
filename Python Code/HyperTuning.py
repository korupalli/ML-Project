def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    dfs = []
    df1 = training_df
    
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        
        for j in [nb_0, nb_1, nb_2, svm_0, svm_1, svm_2]:
            lr_model = nb_0.fit(c_train)
            lr_pred = lr_model.transform(c_test)
            dfs.append(lr_pred)

    for i in range(6):
        df2 = dfs[i].unionAll(dfs[i+1]).unionAll(dfs[i+2]).unionAll(dfs[i+3]).unionAll(dfs[i+4]).select(['id','features','label','group','label_0','label_1','label_2','nb_pred_0'])
        print(df1.count())
        df1 = df1.join(df2, ['id','features','label','group','label_0','label_1','label_2']).count()
    
    return df1