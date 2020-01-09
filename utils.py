# Utils functions
import matplotlib.pyplot as plt
import seaborn as sns
import math


def missing_values(df, col_name):
    """
    input:
    df: dataframe
    col_name: column name in df
    Report missing values (nan) in df columns
    """
    count_null = df[df[col_name].isna()].shape[0]
    count_total = df[col_name].shape[0]
    print(f"\t{col_name}: {(count_null,count_total)} {count_null/count_total}%")
    

def basic_analisis_df(df, df_name):
    """
    input:
    df: dataframe
    df_name: title of report
    Report shape, datatype columns and missing values per df
    """
    print(df_name)
    print(f"(Filas,columnas): \n \t{df.shape}")
    print("columna (type):\n {}".format(''.join(["\t" + col +' ('+  str(df[col].dtype) + ')\n ' for col in df.columns])))
    print("Missing values (na_count , total) percent% \n")
    [missing_values(df,col) for col in df.columns]
    
def graph_df_basics(df, figsize=(15,15)):
    """
    df: dataframe
    figsize: tuple (widht, height) size figure (default (15,15))
    
    Plot each df column using sns.countplot in obect type and distplot in numeric type
    """
    height = math.ceil(df.shape[1]/2)
    ax = plt.figure(figsize=figsize)
    ax.subplots(height,2)
    image_pos = 1
    for col_name, col_serie in df.iteritems():
        plt.subplot(height,2,image_pos)
        plt.title(f"Chart {col_name}")
        if col_serie.dtype == 'object':
            valores_unicos = col_serie.unique().shape[0]
            if valores_unicos < 50:
                sns.countplot(col_serie,palette=sns.color_palette("Paired"));
            else:
                msje = f"Columna {col_name} es tipo object con {valores_unicos} categorias "
                plt.text(.2, .5, msje, fontsize=12)
        else:
            sns.distplot(col_serie.dropna(),color="red",
                 kde=False);
        image_pos +=1
        plt.tight_layout()