def create_columns_for_U(U: list, df, column_name: list[str]):
    for i in U:
        for j in column_name:
            df[j+f"U_{i}"] = df.loc[:, j]
            df[j+f"U_{i}"] *= i**df['ord']
    return df