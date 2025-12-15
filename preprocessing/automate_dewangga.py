import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

raw_data_path = "namadaset_raw/StudentsPerformance.csv"
preprocessed_dir = "preprocessing/namadataset_preprocessing"
os.makedirs(preprocessed_dir, exist_ok=True)
preprocessed_file = os.path.join(preprocessed_dir, "StudentsPerformance_preprocessed.csv")

df = pd.read_csv(raw_data_path)

numeric_cols = ['math score', 'reading score', 'writing score']
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education',
                    'lunch', 'test preparation course']

scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric_scaling", scaler, numeric_cols),
        ("categorical_encoding", encoder, categorical_cols)
    ]
)

processed_array = preprocessor.fit_transform(df)

encoded_cols = preprocessor.named_transformers_['categorical_encoding'].get_feature_names_out(categorical_cols)
all_columns = numeric_cols + list(encoded_cols)

processed_df = pd.DataFrame(processed_array, columns=all_columns)
processed_df.to_csv(preprocessed_file, index=False)
