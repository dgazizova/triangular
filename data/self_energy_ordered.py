import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_for_plot import create_columns_for_U


orders_used = [2, 3, 4, 5]
U_values = [1]

# Replace 'your_file.dat' with the actual path to your DAT file
file_path_1 = f'self_energy_mu_0_kf1.dat'

# Read the DAT file into a Pandas DataFrame
column_names = ['ord', 'U', 'reW', 'imW', 'beta', 'remmu', 'immu', 'kx', 'ky', "resRe", "resRe_err", "resIm",
                "resIm_err"]
df_1 = pd.read_table(file_path_1, delimiter=' ', header=None, names=column_names)

filtered_df_1 = df_1[(df_1["U"].isin(U_values)) & (df_1["ord"].isin(orders_used))]
filtered_df_1 = filtered_df_1.drop(['reW', "U", "immu"], axis=1)
print(filtered_df_1)


U_to_plot = [1, 2, 3, 4]
filtered_df_1 = create_columns_for_U(U_to_plot, filtered_df_1, ["resRe", "resRe_err", "resIm", "resIm_err"])





U = 3
kf_1 = (r"$(\pi / \sqrt{3}, \pi / 2)$")

fig = plt.figure(figsize=(16, 10))
for idx, i in enumerate(orders_used):
    filtered_df_1_by_ord = filtered_df_1[filtered_df_1["ord"] == i]
    print(filtered_df_1_by_ord)
    plt.subplot(2, len(orders_used), idx+1)
    plt.title(f'ord={i}', size=15)
    plt.ylabel(r'Re$\Sigma(k_f, \omega)$', size=15)
    plt.xlabel(r'$i\omega$', size=15)
    plt.errorbar(filtered_df_1_by_ord['imW'], filtered_df_1_by_ord[f'resReU_{U}'], yerr=filtered_df_1_by_ord[f'resRe_errU_{U}'], fmt='.-',
                 label=f"{kf_1}", color="orange")
    plt.legend()

    plt.subplot(2, len(orders_used), idx+1 + len(orders_used))
    plt.title(f'ord={i}', size=15)
    plt.ylabel(r'Im$\Sigma(k_f, \omega)$', size=15)
    plt.xlabel(r'$i\omega$', size=15)
    plt.errorbar(filtered_df_1_by_ord['imW'], filtered_df_1_by_ord[f'resImU_{U}'], yerr=filtered_df_1_by_ord[f'resIm_errU_{U}'], fmt='.-',
                 label=f"{kf_1}")
    plt.legend()


plt.tight_layout()
plt.savefig("/Users/dariagazizova/Desktop/triangular/figure/self_energy_mu0_kf1_ord_U3.pdf", format='pdf')
plt.show()
