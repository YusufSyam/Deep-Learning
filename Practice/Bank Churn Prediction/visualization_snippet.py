# Library untuk pengolahan data
import pandas as pd
import numpy as np
from collections import Counter

# Library untuk visualisasi
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm, colors

# Fungsi untuk mengembalikan daftar kolom kategorik pada dataframe yang diberikan
def categorical_cols(df):
    return df.select_dtypes(object).columns

# Fungsi untuk mengembalikan daftar kolom numerik pada dataframe yang diberikan
def numerical_cols(df):
    return df.select_dtypes(np.number).columns

# Fungsi untuk mengembalikan warna acak dari colormap yang diberikan
def get_random_cmap(choose_cmap='Set2', color_num=2):
    cmap = cm.get_cmap(choose_cmap)
    cmap_list = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    replace = False
    if (color_num > len(cmap_list)):
        print('color is not unique anymore because color_num is bigger than cmap length')
        replace = True

    color_index = np.random.choice(cmap.N, color_num, replace=replace)

    return cmap_list[color_index[0]] if (color_num == 1) else [cmap_list[ci] for ci in color_index]


# Fungsi ini akan menyamakan panjang dari dua dataframe kategorik yang berbeda dengan cara
# Dataframe yang lebih kecil mengambil kategori dari dataframe yang lebih besar, dan mengisi frekuensinya dengan 0
def length_equalize(series1, series2):
    a = pd.DataFrame(series1.sort_index()).reset_index().iloc[::-1]
    b = pd.DataFrame(series2.sort_index()).reset_index().iloc[::-1]

    if (len(a) == len(b)):
        return a, b

    bigger, smaller_temp = (a, b) if (len(a) > len(b)) else (b, a)
    max_iteration = len(bigger.index)

    smaller = bigger.copy()

    j = 0
    for i in range(max_iteration):
        if ((i <= len(smaller_temp.index)) and (bigger.iloc[i, 0] == smaller_temp.iloc[j, 0])):
            smaller.iloc[i, 1] = smaller_temp.iloc[j, 1]
            j += 1
        else:
            smaller.iloc[i, 1] = 0

    return (bigger, smaller) if (bigger.equals(a)) else (smaller, bigger)

def sort_by_target(a, b, target_count_column):
    total= a.iloc[:,1]+b.iloc[:,1]

    a[target_count_column]= a.iloc[:,1]/total
    b[target_count_column]= b.iloc[:,1]/total

    b= b.sort_values(by=target_count_column, ascending=False).reset_index(drop=True)

    a= a.set_index('index')
    a= a.reindex(index=b['index'])
    a= a.reset_index()

    return a, b


def multiple_bar_plot(df, class1, class2, label1, label2, title, normalize):
    # Mendapatkan daftar kategori pada kolom
    unique_values = df.unique()

    # Mendefinisikan xticks
    x_indexes = np.arange(len(unique_values))

    # Mendefinisikan lebar setiap sub-bar
    width = 0.3

    # Memanggil fungsi value_counts() pada kedua kelas untuk mendapatkn kategori dan frekuensinya
    class1 = class1.value_counts()
    class2 = class2.value_counts()

    # Karena pada multiple bar length dari data harus sama, maka kita memanggil fungsi length_equalize yang dibuat sebelumnya
    class1, class2 = length_equalize(class1, class2)

    # Mendapatkan 2 warna acak untuk masing-masing bar
    color1, color2 = get_random_cmap('Set2', 2)

    class_bar_height = 'bar_height'

    class1[class_bar_height] = class1.iloc[:, 1]
    class2[class_bar_height] = class2.iloc[:, 1]

    if (normalize):
        class1, class2 = sort_by_target(class1, class2, class_bar_height)

    # Melakukan plot dalam bentuk multiple bar
    plt.bar(x_indexes - (width / 2), class1[class_bar_height], width, label=label1, color=color1)
    plt.bar(x_indexes + (width / 2), class2[class_bar_height], width, label=label2, color=color2)

    # Menyetel atribut-atribut dari plt
    plt.xticks(ticks=x_indexes, labels=class1.iloc[:, 0], rotation=45)
    plt.title(f'Bar chart of {title}', size=18, pad=20)

    # Menyetel atribut-atribut dari plt
    plt.legend()
    plt.ylabel('Count')

    if (normalize):
        plt.xticks(fontsize=10)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        ax = plt.gca()

        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1%}', (x + width / 2, y + height * 1.05), ha='center')

    plt.tight_layout()
    plt.show()


def multiple_hist_plot(df, class1, class2, label1, label2, title):
    # Mendapatkan 2 warna acak untuk masing-masing histogram
    color1, color2 = get_random_cmap('Set2', 2)

    # Mendefinisikan bins (jumlah bar pada histogram)
    bins = 15

    # Melakukan plot dalam bentuk histogram dengan seaborn 2x
    sns.histplot(class1, kde=True, color=color1, bins=bins, alpha=0.75, label=label1)
    sns.histplot(class2, kde=True, color=color2, bins=bins, alpha=0.75, label=label2)

    # Menyetel atribut-atribut dari plt
    plt.title(f'Histogram of {title}', size=18)

    # Menyetel atribut-atribut dari plt
    plt.legend()
    plt.ylabel('Count (log)')
    plt.yscale('log')
    plt.tight_layout()

    plt.show()


# Meskipun kita akan membandingkan data dari dua class pada perulangan, sebaiknya kita mendefinisikan label di luar perulangan

def binary_dataframe_comparison(df, target_column, figsize=(16, 6), bar_normalize=True):
    label1 = df[target_column].unique()[0]
    label2 = df[target_column].unique()[1]

    df_columns = df.columns[:-1]
    df_categorical_cols = categorical_cols(df[:-1])

    for i in df_columns:
        plt.figure(figsize=figsize)

        # Membagi kolom menjadi dua dataframe
        # Dataframe class1 yaitu kolom dengan salary_per_year <= 50 k begitu pula sebaliknya
        class1 = df[df[target_column] == label1][i]
        class2 = df[df[target_column] == label2][i]

        # Memberi kondisi jika kolom berupa kolom kategorik maka akan dilakukan plot dalam bentuk multiple bar
        if (i in df_categorical_cols):
            multiple_bar_plot(df[i], class1, class2, label1, label2, i, bar_normalize)

        # Jika kolom bukan merupakan kolom kategorik maka plot akan diisi dengan 2 histogram
        else:
            multiple_hist_plot(df[i], class1, class2, label1, label2, i)

