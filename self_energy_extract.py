import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_for_plot import create_columns_for_U


orders_used = [2, 3, 4, 5]
U_values = [1]

# Replace 'your_file.dat' with the actual path to your DAT file
kf = [1, 2]
file_path_1 = f'/Users/daria/Desktop/trangular/self_energy/o12345/kf1/ord_U_results.dat'
file_path_2 = f'/Users/daria/Desktop/trangular/self_energy/o12345/kf2/ord_U_results.dat'

# Read the DAT file into a Pandas DataFrame
column_names = ['ord', 'U', 'reW', 'imW', 'beta', 'remmu', 'immu', 'kx', 'ky', "resRe", "resRe_err", "resIm",
                "resIm_err"]
df_1 = pd.read_table(file_path_1, delimiter=' ', header=None, names=column_names)
df_2 = pd.read_table(file_path_2, delimiter=' ', header=None, names=column_names)

filtered_df_1 = df_1[(df_1["U"].isin(U_values)) & (df_1["ord"].isin(orders_used))]
filtered_df_1 = filtered_df_1.drop(['reW', "U", "immu"], axis=1)

# print(filtered_df_1)

filtered_df_2 = df_2[(df_2["U"].isin(U_values)) & (df_2["ord"].isin(orders_used))]
filtered_df_2 = filtered_df_2.drop(['reW', "U", "immu"], axis=1)
# print(filtered_df_2)

"""plot for U values"""
U_to_plot = [1, 2, 3, 4]
filtered_df_1 = create_columns_for_U(U_to_plot, filtered_df_1, ["resRe", "resRe_err", "resIm", "resIm_err"])
print(filtered_df_1[["ord", "resReU_1", "resReU_2"]])
filtered_df_2 = create_columns_for_U(U_to_plot, filtered_df_2, ["resRe", "resRe_err", "resIm", "resIm_err"])

summed_df_1 = filtered_df_1.groupby("imW").sum().reset_index()
summed_df_1 = summed_df_1.drop(['ord', 'beta', "remmu"], axis=1)

summed_df_2 = filtered_df_2.groupby("imW").sum().reset_index()
summed_df_2 = summed_df_2.drop(['ord', 'beta', "remmu"], axis=1)


print(summed_df_1)
print(summed_df_2)


for i in U_to_plot:
    plt.subplot(2, 4, i)
    plt.title(f'U/t = {i}')
    plt.ylabel(r'Re$\Sigma(k_f, \omega)$')
    plt.xlabel(r'$i\omega$')
    plt.errorbar(summed_df_1['imW'], summed_df_1[f'resReU_{i}'], yerr=summed_df_1[f'resRe_errU_{i}'], fmt='.-',
                 label=r"$kf_1$")
    plt.errorbar(summed_df_2['imW'], summed_df_2[f'resReU_{i}'], yerr=summed_df_2[f'resRe_errU_{i}'], fmt='.-',
                 label=r"$kf_2$")
    plt.legend()

    plt.subplot(2, 4, i + 4)
    plt.title(f'U/t = {i}')
    plt.ylabel(r'Im$\Sigma(k_f, \omega)$')
    plt.xlabel(r'$i\omega$')
    plt.errorbar(summed_df_1['imW'], summed_df_1[f'resImU_{i}'], yerr=summed_df_1[f'resIm_errU_{i}'], fmt='.-',
                 label=r"$kf_1$")
    plt.errorbar(summed_df_2['imW'], summed_df_2[f'resImU_{i}'], yerr=summed_df_2[f'resIm_errU_{i}'], fmt='.-',
                 label=r"$kf_2$")
    plt.legend()


plt.show()
