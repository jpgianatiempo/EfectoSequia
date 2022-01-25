import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

# Cargo los datos de estimaciones de la Bolsa de Cereales
# https://www.bolsadecereales.com/download/informes/documento2/121
data = pd.read_csv("./Data/historico_pas_datasets.csv", encoding="latin-1", sep=";")

# Filtro para soja y maíz
data.query('Cultivo == "Soja" | Cultivo == "Maiz"', inplace = True)

# Genero porcentaje de area perdida
data["Porcentaje_Area_Perdida"] = data["Perdído(Ha)"] / data["Sembrado (Ha)"]

# Cargo los datos de ENSO
# https://psl.noaa.gov/enso/mei/data/meiv2.data
enso = pd.read_excel("./Data/ENSO.xlsx")

# Armo nueva columna con los valores absolutos para el tamaño de los puntos en los gráficos
enso = enso.assign(JF_ABS = (enso["JF"].abs()))

#join con el resto de la data
data = data.merge(enso.filter(['Campaña', 'ENSO JF']), how='left', on='Campaña')

# Gráficos
# Seteo el estilo de los gráficos
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

# Paleta hecha a mano
palette = {"Niña":"tab:red",
           "Neutral":"tab:blue", 
           "Niño":"tab:green"}

#Gráfico Campaña y ENSO
campañas = list(["2000/2001","2001/2002","2002/2003","2003/2004","2004/2005","2005/2006","2006/2007","2007/2008","2008/2009","2009/2010","2010/2011","2011/2012","2012/2013","2013/2014","2014/2015","2015/2016","2016/2017","2017/2018","2018/2019","2019/2020","2020/2021","2021/2022 (P)"])
graph = sns.relplot(x="JF", y="Campaña", hue="ENSO JF", size="JF_ABS",
            sizes=(20, 400), palette=palette, aspect=1.5,
            height=8, data=enso[enso['Campaña'].isin(campañas)]).set(title="ENSO durante enero y febrero",ylabel='Campaña', xlabel='Índice Multivariado ENSO (MEI)',xlim=(-2, 2.5))._legend.set_title("ENSO")
plt.axvline(x=0.5, color='g', linestyle='--')
plt.axvline(x=-0.5, color='r', linestyle='--')
plt.savefig('./Graphs/ENSO.png')

# Distribución de rendimiento de soja por ENSO
sns.displot(data=data.query('Cultivo == "Soja"'), x='Rinde(qq/Ha)', hue='ENSO JF', kind='kde', fill=True, palette=sns.color_palette('bright')[:3], height=5, aspect=1.5).set(title="Soja: Distribución de rendimientos",xlabel='Rinde qq/ha', ylabel='Densidad')._legend.set_title("ENSO")
plt.savefig('./Graphs/Rinde soja por enso.png')

# Distribución de rendimiento de maíz por ENSO
sns.displot(data=data.query('Cultivo == "Maiz"'), x='Rinde(qq/Ha)', hue='ENSO JF', kind='kde', fill=True, palette=sns.color_palette('bright')[:3], height=5, aspect=1.5).set(title="Maíz: Distribución de rendimientos",xlabel='Rinde qq/ha', ylabel='Densidad')._legend.set_title("ENSO")
plt.savefig('./Graphs/Rinde maíz por enso.png')

# Distribución del área perdida
sns.displot(data=data.query('Cultivo == "Soja"'), x='Porcentaje_Area_Perdida', hue='ENSO JF', kind='kde', fill=True, palette=sns.color_palette('bright')[:3], height=5, aspect=1.5).set(title="Soja: Distribución del área perdida",xlabel='Área perdida (%)', ylabel='Densidad')._legend.set_title("ENSO")
plt.savefig('./Graphs/Area perdida soja por enso.png')

# Distribución del área perdida
sns.displot(data=data.query('Cultivo == "Maiz"'), x='Porcentaje_Area_Perdida', hue='ENSO JF', kind='kde', fill=True, palette=sns.color_palette('bright')[:3], height=5, aspect=1.5).set(title="Maíz: Distribución del área perdida",xlabel='Área perdida (%)', ylabel='Densidad')._legend.set_title("ENSO")
plt.savefig('./Graphs/Area perdida maíz por enso.png')



# Filtro las campañas que son Niña
Niña = data[data["ENSO JF"]=="Niña"]

# Agrupo por zona y cultivo -- calculo min, max, media del % area perdida y rinde
Niña = Niña.groupby(["Zona","Cultivo"]).agg({'Rinde(qq/Ha)': ['mean', 'min', 'max'],
                                                   'Porcentaje_Area_Perdida': ['mean', 'min', 'max']}).reset_index()

# Saco la agrupacion de los nombres en las columna
Niña.columns = ['_'.join(col) for col in Niña.columns]

# Cargo datos del área sembra de la campaña 2021/22 del último PAS de la bolsa de Cereales
a2121 = pd.read_excel("./Data/area2122.xlsx")

#junto con el dataset de área 2021/22
Niña = Niña.merge(a2121, how="left", on=["Zona_","Cultivo_"])

#calculo los 3 escenarios
Niña = Niña.assign(
    Area_cosechada_mean = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_mean']),
    Area_cosechada_min = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_max']),
    Area_cosechada_max = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_min']),
    Produccion_mean = lambda x: x['Rinde(qq/Ha)_mean'] * x['Area_cosechada_mean']/10,
    Produccion_min = lambda x: x['Rinde(qq/Ha)_min'] * x['Area_cosechada_min']/10,
    Produccion_max = lambda x: x['Rinde(qq/Ha)_max'] * x['Area_cosechada_max']/10
)

# Saco el total país
Niña = Niña[Niña["Zona_"]!="TOTAL"]

# Selecciono las columnas que me interesan
Nacional = Niña.filter(["Zona_","Cultivo_", "Produccion_mean","Produccion_min","Produccion_max"])

# Calculo produccion total nacional de los 3 escenarios
Nacional = Nacional.groupby(["Cultivo_"]).agg({'Produccion_mean': 'sum',
                                                   'Produccion_min': 'sum',
                                                   'Produccion_max': 'sum'}).assign(
                                                           Produccion_mean = lambda x: x['Produccion_mean']/1000000,
                                                           Produccion_min = lambda x: x['Produccion_min']/1000000,
                                                           Produccion_max = lambda x: x['Produccion_max']/1000000
                                                   )

Niña.to_excel("./Data/Escenarios_niña_provincias.xlsx")
Nacional.to_excel("./Data/Escenarios_niña_nacional.xlsx")




#genero año neutral promedio
Neutral = data[data["ENSO JF"]=="Neutral"]
Neutral = Neutral.groupby(["Zona","Cultivo"]).agg({'Rinde(qq/Ha)': ['mean', 'min', 'max'],
                                                   'Porcentaje_Area_Perdida': ['mean', 'min', 'max']}).reset_index()

#saco la agrupacion de la columna
Neutral.columns = ['_'.join(col) for col in Neutral.columns]

#junto con el dataset de área 2021/22
a2121 = pd.read_excel("./Data/area2122.xlsx")

Neutral = Neutral.merge(a2121, how="left", on=["Zona_","Cultivo_"])

#calculo los 3 escenarios
Neutral = Neutral.assign(
    Area_cosechada_mean = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_mean']),
    Area_cosechada_min = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_max']),
    Area_cosechada_max = lambda x: x['Area_sembrada'] * (1-x['Porcentaje_Area_Perdida_min']),
    Produccion_mean = lambda x: x['Rinde(qq/Ha)_mean'] * x['Area_cosechada_mean']/10,
    Produccion_min = lambda x: x['Rinde(qq/Ha)_min'] * x['Area_cosechada_min']/10,
    Produccion_max = lambda x: x['Rinde(qq/Ha)_max'] * x['Area_cosechada_max']/10
)

Neutral.to_excel("./Escenarios_neutral_max.xlsx")




