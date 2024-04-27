import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller

st.set_option('deprecation.showPyplotGlobalUse', False)


def tema_estadisticas_descriptivas():
    st.title("Estadísticas Descriptivas")

    # Celda de Markdown para el título y detalles de la aplicación
    st.header('Análisis de Salarios')
    st.write("""
    Esta aplicación realiza un análisis exploratorio de salarios simulados.
    """)

    # Generar salarios aleatorios 
    np.random.seed(1900)
    salarios = np.random.normal(loc=50000, scale=15000, size=250)
    salarios = np.abs(salarios)
    salarios = np.round(salarios, decimals=-3)

    # Calcular estadísticas descriptivas
    media = np.mean(salarios)
    mediana = np.median(salarios)
    varianza = np.var(salarios)
    desviacion_estandar = np.std(salarios)
    #moda_result = stats.mode(salarios)
    #moda = moda_result.mode[0] if moda_result.count[0] > 1 else "No hay moda"
    rango = np.max(salarios) - np.min(salarios)

    # Celda de Markdown para mostrar las estadísticas descriptivas
    st.write(f"**Media:** {media}")
    st.write(f"**Mediana:** {mediana}")
    #st.write(f"**Moda:** {moda}")
    st.write(f"**Varianza:** {varianza}")
    st.write(f"**Desviación Estándar:** {desviacion_estandar}")
    st.write(f"**Rango:** {rango}")

    st.header('Análisis de las Estadísticas Descriptivas de los Datos')
    st.write("""
    La media y la mediana están relativamente cerca, esto nos indica que la distribución de los salarios no está muy sesgada.
    La varianza y la desviación estándar son un poco más altas, esto nos indica que hay una mayor dispersión en los salarios.

    Los trabajadores en este conjunto de datos ganan un salario promedio anual de alrededor de $50,000.
    Hay una gran variabilidad en los salarios, con algunos trabajadores ganando mucho más que otros.
    La distribución de los salarios en este conjunto de datos parece estar un poco más concentrada en el extremo inferior.
    """)
    # Mostrar histograma
    st.subheader('Histograma de Salarios')
    hist_fig, ax = plt.subplots()  # Crear la figura
    ax.hist(salarios, bins=8, color='blue', edgecolor='black')
    ax.set_title('Histograma de Salarios')
    ax.set_xlabel('Salarios')
    ax.set_ylabel('Frecuencia')
    st.pyplot(hist_fig)  # Pasar la figura a st.pyplot()

    # Mostrar diagrama de bigotes (boxplot)
    st.subheader('Boxplot de Salarios')
    boxplot_fig, ax = plt.subplots()  # Crear la figura
    ax.boxplot(salarios)
    ax.set_title('Boxplot de Salarios')
    ax.set_ylabel('Salarios')
    st.pyplot(boxplot_fig)  # Pasar la figura a st.pyplot()

    # # Mostrar diagrama de pastel de la media y la moda
    # st.subheader('Diagrama de Pastel de Media vs Moda')
    # labels = ['Media', 'Moda']
    # sizes = [media, moda_result.mode[0]] if moda != "No hay moda" else [media, 0]
    # sizes = [size for size in sizes if isinstance(size, (int, float))]  # Filtrar

    # colors = ['gold', 'yellowgreen']
    # plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    # plt.axis('equal')
    # plt.title('Media vs Moda')
    # st.pyplot()

    # Mostrar diagrama de pastel de los salarios menores y mayores a 50k
    st.subheader('Diagrama de Pastel de Salarios Menos de 50k vs Más de 50k')
    salarios_menor_50k = salarios[salarios < 50000]
    salarios_mayor_50k = salarios[salarios >= 50000]
    sizes = [len(salarios_menor_50k), len(salarios_mayor_50k)]
    labels = ['Menos de 50k', 'Más de 50k']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    pie_fig2, ax = plt.subplots()  # Crear la figura
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(pie_fig2)  # Pasar la figura a st.pyplot()

    st.write("""
    Con estos gráficos de pastel logramos observ ar que la mayoría de los trabajadores en este conjunto de datos ganan menos de $50,000 al año.
    Esto sugiere que la mayoría de los trabajadores en este conjunto de datos ganan salarios relativamente bajos.

    Por otro lado, la distribución de los salarios en este conjunto de datos parece estar un poco más concentrada en el extremo inferior.
    Esto sugiere que hay una mayor concentración de trabajadores en el extremo inferior de los salarios.De esta foram al hacer una comparación entre la media y la moda, logramos darnos cuenta que la media de los salarios es de 55.6% y la moda es de 44.4%, lo que nos indica que la mayoría de los trabajadores en este conjunto de datos ganan menos de $50,000 al año.

    Además, es importante notar que aunque la media de los salarios es de 55.6%, esto no necesariamente significa que todos los trabajadores estén ganando esta cantidad. La media puede ser influenciada por valores extremos, por lo que puede que haya una pequeña cantidad de trabajadores que ganan mucho más que el resto, elevando la media.

    En contraste, la moda, que es el valor más frecuente, nos indica que el salario más comúnmente ganado es menos de $50,000 al año. Esto refuerza la idea de que la mayoría de los trabajadores en este conjunto de datos están en el extremo inferior de la escala salarial.

    Estos hallazgos pueden tener implicaciones significativas para las políticas de compensación y beneficios de la empresa. Por ejemplo, si la empresa desea retener a sus trabajadores, podría ser necesario revisar las estructuras salariales y considerar aumentos o beneficios adicionales para aquellos en el extremo inferior.
    """)

    # Mostrar diagrama de violín
    st.subheader('Violinplot de Salarios')
    violin_fig, ax = plt.subplots()  # Crear la figura
    ax.violinplot(salarios, showmedians=True)
    ax.set_title('Violinplot de Salarios')
    ax.set_ylabel('Salarios')
    st.pyplot(violin_fig)  # Pasar la figura a st.pyplot()

    # Mostrar dispersión de los salarios
    st.subheader('Dispersión de Salarios')
    scatter_fig, ax = plt.subplots()  # Crear la figura
    ax.plot(salarios, 'o', label='Salarios')
    ax.axhline(media, color='r', linestyle='-', label=f'Media: {media:.2f}')
    ax.axhline(media + desviacion_estandar, color='g', linestyle='--', label=f'+1 Desv. Est.: {media + desviacion_estandar:.2f}')
    ax.axhline(media - desviacion_estandar, color='g', linestyle='--', label=f'-1 Desv. Est.: {media - desviacion_estandar:.2f}')
    ax.set_title('Dispersión de Salarios')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Salarios')
    ax.legend()
    st.pyplot(scatter_fig)  # Pasar la figura a st.pyplot()

    st.write("""
    La distribución de salarios es asimétrica, con una mayor concentración de salarios en el rango de $40,000 a $60,000, como indica la mediana en $49,000 y la media en $50,144.
    Hay algunos salarios más altos que se extienden hacia la derecha, lo que sugiere la presencia de algunos valores atípicos o salarios más altos en el conjunto de datos.
    La mediana y la media están relativamente cerca, lo que nos indica que la distribución de salarios está más o menos centrada alrededor de estos valores. 
    La varianza y la desviación estándar son indicadores de dispersión. La varianza es alta, lo que sugiere una dispersión considerable en los salarios con respecto a la media. 
    La distribución de salarios parece ser relativamente concentrada alrededor de un valor central, pero con una dispersión considerable.
    """)

def tema_patrones():
    import streamlit as st
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    st.title("Patrones en Series Temporales")
    #!/usr/bin/env python
    # coding: utf-8

    st.header("Problema de identificación de patrones en datos de ventas mensuales")

    # Definir productos y años
    productos = ['iPhone 11', 'iPhone 12', 'iPhone 13']
    años = 3

    # Función para generar ventas mensuales
    def generar_ventas_mensuales(productos, años):
        fecha_inicio = datetime(2021, 1, 1)
        fecha_fin = fecha_inicio + timedelta(days=365 * años)
        ventas = {'Fecha': [], 'Producto': [], 'Ventas': []}

        for producto in productos:
            fecha = fecha_inicio
            while fecha < fecha_fin:
                ventas['Fecha'].append(fecha)
                ventas['Producto'].append(producto)
                ventas['Ventas'].append(random.randint(0,300 ))  # Ventas aleatorias entre 100 y 1000
                fecha += timedelta(days=30)  # Incremento de un mes

        return pd.DataFrame(ventas)

    # Generar datos de ventas
    datos_ventas = generar_ventas_mensuales(productos, años)

    # Mostrar los primeros registros
    print(datos_ventas.head())

    print(datos_ventas)

    st.write("""
            En la primera parte del código lo que se hizo fue generar datos simulados que representarán las ventas mensuales de tres productos 
            (iPhone 11, iPhone 12 y iPhone 13) a lo largo de un período de tiempo.""")
    
    # Calcular estadisticas descriptivas
    estadisticas = datos_ventas.groupby('Producto')['Ventas'].describe()
    print(estadisticas)

    st.write("""
            El primer análisis que se realizó fue el de calcular las estadísticas descriptivas de las ventas mensuales de cada producto. Para ello, se utilizó la función `describe()` de la librería `pandas`. Los datos que nos arrojó esta función se pueden interpretar de la siguiente manera:

            El promedio de ventas del iPhone 11 fue de aproximadamente 124 unidades, con una desviación estándar aproximadamente 91 unidades, lo que indica una dispersión relativamente alta en las ventas.De la misma manera, el número mínimo de ventas registrado para el iPhone 11 fue de 5 unidades.El 25% más bajo de las ventas fue de 46 unidades o menos. La mediana de las ventas fue de 94 unidades, lo que significa que el 50% de las ventas estuvieron por debajo de este valor. El 75% más bajo de las ventas fue de 182 unidades o menos y el número máximo de ventas fue de 300 unidades.

            El promedio de ventas del iPhone 12 fue de aproximadamente 173 unidades, su desviación estándar de las ventas del iPhone 12 fue de aproximadamente 91 unidades, similar a la del iPhone 11. El número mínimo de ventas registrado fue de 11 unidades. El 25% más bajo de las ventas fue de 106 unidades o menos. La mediana de las ventas fue de 174 unidades. El 75% más bajo de las ventas fue de 254 unidades o menos.El número máximo de ventas registrado fue de 295 unidades.

            El promedio de ventas del iPhone 13 fue de aproximadamente 148 unidades, su desviación estándar de las ventas del iPhone 13 fue de aproximadamente 70 unidades, la más baja entre los tres modelos. El número mínimo de ventas fue de 0 unidades. El 25% más bajo de las ventas fue de 107 unidades o menos. La mediana de las ventas fue de 153 unidades. El 75% más bajo de las ventas fue de 202 unidades o menos y el número máximo de ventas registrado fue de 261 unidades.
             
             """)
    
    # Mostrar la cantidad de productos que se vendieron por año 

    datos_ventas['Año'] = datos_ventas['Fecha'].dt.year
    productos_vendidos_por_año = datos_ventas.groupby(['Producto', 'Año'])['Ventas'].sum()
    print(productos_vendidos_por_año)


    # Histograma de las ventas por producto en cada año
    plt.figure()  # Crear la figura
    datos_ventas['Año'] = datos_ventas['Fecha'].dt.year
    datos_ventas.groupby(['Producto', 'Año'])['Ventas'].sum().unstack().plot(kind='bar', stacked=True, title='Ventas por producto y año')
    fig1 = plt.gcf()  # Obtener la figura actual
    st.pyplot(fig1)

    st.write("""
            Con ayuda de la función `plot()` de la librería `pandas`, se generaron gráficos
            de barras para visualizar las ventas anuales de cada producto. En estos 
            gráficos, se puede observar que las ventas del iPhone 12 fueron las más 
            altas en general, seguidas por las del iPhone 13 y luego por las del iPhone 11.
            Además, se puede observar que las ventas del iPhone 12 aumentaron a lo largo del 
            tiempo, mientras que las ventas del iPhone 11 disminuyeron.""")

    # Mostrar las ventas de cada producto por mes
    plt.figure()  # Crear la figura
    datos_ventas['Mes'] = datos_ventas['Fecha'].dt.month
    ventas_por_mes = datos_ventas.groupby(['Producto', 'Mes'])['Ventas'].sum()
    ventas_por_mes.unstack().plot(kind='bar', title='Ventas por producto y mes')
    fig2 = plt.gcf()  # Obtener la figura actual
    st.pyplot(fig2)

    st.write("""
            El gráfico que se mostró arriba, nos muestra de manera más especifica las 
            ventas mensuales de cada producto a lo largo del tiempo. En este gráfico, 
            se puede observar que las ventas del iPhone 12 fueron las más altas en general,
            seguidas por las del iPhone 13 y luego por las del iPhone 11. De igual manera, podemos observar que el mes con las ventas más altas para el iPhone 12 y el el iPhone 11 fue el mes de enero, mientras que para el iPhone 13 fue el mes de diciembre, con esto podemos deducir que existe una tendencia de comprar en esos meses por cuestuines de temporada, en este caso, las fiestas de fin de año, que es muy comuún por la costumbre de regalar en estas fechas.
            """)
    
    # Mostrar las ventas del iphone 11 por mes, del año 2021
    ventas_iphone_11_2021 = datos_ventas[(datos_ventas['Producto'] == 'iPhone 11') & (datos_ventas['Fecha'].dt.year == 2021)]
    print(ventas_iphone_11_2021)

    # Graficarlo en un gráfico de barras
    plt.figure()  # Crear la figura
    ventas_iphone_11_2021.groupby('Mes')['Ventas'].sum().plot(kind='bar', title='Ventas del iPhone 11 en 2021')
    fig3 = plt.gcf()  # Obtener la figura actual
    st.pyplot(fig3)

    # Mostrar las ventas del iphone 12 por mes, del año 2021
    ventas_iphone_12_2021 = datos_ventas[(datos_ventas['Producto'] == 'iPhone 12') & (datos_ventas['Fecha'].dt.year == 2021)]
    print(ventas_iphone_12_2021)

    # Graficarlo en un gráfico de barras
    plt.figure()  # Crear la figura
    ventas_iphone_12_2021.groupby('Mes')['Ventas'].sum().plot(kind='bar', title='Ventas del iPhone 12 en 2021')
    fig4 = plt.gcf()  # Obtener la figura actual
    st.pyplot(fig4)

    # Mostrar las ventas del iphone 13 por mes, del año 2021
    ventas_iphone_13_2021 = datos_ventas[(datos_ventas['Producto'] == 'iPhone 13') & (datos_ventas['Fecha'].dt.year == 2021)]
    print(ventas_iphone_13_2021)

    # Graficarlo en un gráfico de barras
    plt.figure()  # Crear la figura
    ventas_iphone_13_2021.groupby('Mes')['Ventas'].sum().plot(kind='bar', title='Ventas del iPhone 13 en 2021')
    fig5 = plt.gcf()  # Obtener la figura actual
    st.pyplot(fig5)

    st.write("""En esos gráficos se mostró de manera más especifica las ventas mensuales 
             de cada producto en el año 2021. En este gráfico, se puede observar que las 
             ventas del iPhone 11 fueron las más altas en general, seguidas por las del 
             iPhone 13 y luego por las del iPhone 12. De igual manera, podemos observar 
             que el mes con las ventas más altas para el iPhone 12 fue el mes de enero, 
             mientras que para el iPhone 13 fue el mes de mayo.""")

    # Gráfico de pastel de las ventas por producto en el último año
    # Crear una nueva figura
    plt.figure()

    # Filtrar los datos para el último año
    datos_ventas['Año'] = datos_ventas['Fecha'].dt.year
    ventas_ultimo_año = datos_ventas[datos_ventas['Año'] == datos_ventas['Año'].max()]

    # Graficar el gráfico de pastel de las ventas por producto en el último año
    ventas_por_producto_ultimo_año = ventas_ultimo_año.groupby('Producto')['Ventas'].sum()
    ventas_por_producto_ultimo_año.plot(kind='pie', autopct='%1.1f%%', title='Ventas por producto en el último año')

    # Obtener la figura actual
    fig6 = plt.gcf()

    # Mostrar la figura en Streamlit
    st.pyplot(fig6)


    st.write("""Con ayuda del gráfico de Pastel, se mostró la distribución de las ventas 
             de cada producto en el último año. En este gráfico, se puede observar que 
             las ventas del iPhone 11 representaron el 21% del total de ventas, las del 
             iPhone 12 representaron el 40.1% y las del iPhone 13 representaron el 38.9%. 
             Por lo tanto, se puede concluir que las ventas del iPhone 12 fueron las más 
             altas en general, seguidas por las del iPhone 13 y luego por las del iPhone 11.
              Lo que nos podría indicar que el iPhone 12 es el producto que se ajusta al 
             presupuesto de la mayoría de las personas y no es tan caro como el iPhone 13, 
             pero tampoco es tan antiguo como el iPhone 11.""")

    # Calcular la media y la desviación estándar
    media = np.mean(datos_ventas['Ventas'])
    desviacion_estandar = np.std(datos_ventas['Ventas'], ddof=1)  # ddof=1 para usar la fórmula de muestra

    # Crear la gráfica de dispersión
    plt.figure(figsize=(10, 6))
    for producto in productos:
        plt.scatter(datos_ventas[datos_ventas['Producto'] == producto]['Fecha'], 
                    datos_ventas[datos_ventas['Producto'] == producto]['Ventas'],
                    label=producto)

    # Añadir líneas para la media y +/- 1 desviación estándar
    plt.axhline(media, color='r', linestyle='-', label=f'Media: {media:.2f}')
    plt.axhline(media + desviacion_estandar, color='g', linestyle='--',
                label=f'+1 Desv. Est.: {media + desviacion_estandar:.2f}')
    plt.axhline(media - desviacion_estandar, color='g', linestyle='--',
                label=f'-1 Desv. Est.: {media - desviacion_estandar:.2f}')

    # Añadir título y etiquetas
    plt.title('Ventas Mensuales por Producto')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.grid(True)

    # Rotar etiquetas del eje x para mayor claridad
    plt.xticks(rotation=45)

    # Mostrar la gráfica
    plt.tight_layout()
    st.pyplot()

    # Realizar descomposición estacional para identificar tendencias, estacionalidad y residuos
    for producto in productos:
        print(f"Descomposición estacional para {producto}:")
        result = seasonal_decompose(datos_ventas[datos_ventas['Producto'] == producto]['Ventas'], model='additive', period=12)
        result.plot()
        st.pyplot()

    st.header("Análisis de los datos")
    st.write("""
            A contiuación analizaremos los patrones de ventas de los productos iPhone 11, iPhone 12 y iPhone 13 a 
            lo largo del tiempo. Para ello, utilizaremos seasonal_decompose que es una función de la librería de 
            Python statsmodels, que se utiliza para descomponer series temporales en sus componentes principales: 
            tendencia, estacionalidad y residuos. Con este tipo de análisis será útil para entender mejor los patrones 
            subyacentes en los datos temporales y mejorar los modelos de predicción. 
            
            En el primer gráfico, se muestra la descomposición de las ventas mensuales del iPhone 11. En este gráfico, 
            se puede observar que la tendencia de las ventas del iPhone 11 fue relativamente estable en un periodo muy 
            corto del tiempo, con un notable descenso en las ventas. La estacionalidad de las ventas del iPhone 11 fue 
            más pronunciada, con picos en los meses de enero y diciembre. Los residuos de las ventas del iPhone 11 
            fueron relativamente grendes, lo que indica que existe la presencia de otros patrones o posibles anomalías.
            
            En el segundo gráfico, se muestra la descomposición de las ventas mensuales del iPhone 12 en sus componentes 
            principales: tendencia, estacionalidad y residuos. En este gráfico, se puede observar que la tendencia de 
            las ventas del iPhone 12 fue gradualmente aumento a lo largo del tiempo, con un ligero descenso a finales 
            del año. La estacionalidad de las ventas del iPhone 12 fue más pronunciada, con picos wn todos los meses. 
            Los residuos de las ventas del iPhone 12 fueron relativamente grendes, lo que indica que existe la presencia 
            de otros patrones o posibles anomalías.
            
            En el tercer gráfico, se muestra la descomposición de las ventas mensuales del iPhone 13 en sus componentes 
            principales: tendencia, estacionalidad y residuos. En este gráfico, se puede observar que la tendencia de las 
            ventas del iPhone 13 no fue relativamente estable a lo largo del tiempo, ya que sus centas empezaron con un 
            pequeño descenso y despues con un pequeño aumento, hasta que a finales del año se observa como gradualmente 
            ascienden sus ventas. La estacionalidad de las ventas del iPhone 13 fue más pronunciada, con picos en los 
            meses de enero, matyo y diciembre. Los residuos de las ventas del iPhone 13 fueron relativamente grendes, 
            lo que indica que existe la presencia de otros patrones o posibles anomalías.
            
            En conclusión, de acuerdo con la descomposición de las ventas, en la parte de los residuos, podríamos decir 
            que existe la presencia de otros patrones o posibles anomalías, lo que nos podría indicar que los datos no 
            son estacionarios y que podrían existir otros factores que influyen en las ventas de los productos.
            """)

def tema_anomalias():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    import random
    from sklearn.ensemble import IsolationForest

    st.title("Anomalías en Series Temporales")

    st.header("Problema para la detección de anomalías en sensores de temperatura en una planta de manufactura")
    
    # Función para generar temperatura normal
    def generate_normal_temperature():
        return round(random.uniform(18, 30), 2)

    # Función para generar temperatura anómala
    def generate_anomalous_temperature():
        return round(random.uniform(5, 40), 2)

    # Función para generar fecha
    def generate_date(start_date, end_date):
        return start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))

    # Función para generar datos de temperatura
    def generate_temperature_data(start_date, end_date, anomaly_rate):
        current_date = start_date
        temperature_data = []

        while current_date < end_date:
            if random.random() < anomaly_rate:
                temperature = generate_anomalous_temperature()
            else:
                temperature = generate_normal_temperature()
            temperature_data.append((current_date.strftime("%m-%d"), temperature))
            current_date += datetime.timedelta(days=1)

        return temperature_data

    # Definir rango de fechas
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    anomaly_rate = 0.05

    # Generar datos de temperatura simulados
    temperature_data = generate_temperature_data(start_date, end_date, anomaly_rate)

    st.write("""En esta parte de código generamos los datos simulados que repesentan las lecturas de un sensor de temperatura. Para ello, se generan datos de temperatura normal en un rango de 18 a 30 grados centígrados. Después, se generan datos de temperatura anómalos en un rango de 5 a 40 grados centígrados, ya que esto podría ser una temperatura extremadamente alta o baja, es importante mencionar que definimos un un 5% de las lecturas como anómalas. Finalmente, se mezclan los datos normales y anómalos para formar el conjunto de datos de entrenamiento.""")
    
    # Convertir los datos a DataFrame
    df_temperature = pd.DataFrame(temperature_data, columns=["Fecha", "Temperatura"])

    # Mostrar resumen estadístico
    st.subheader('Resumen Estadístico de las Lecturas de Temperatura')
    st.write(df_temperature.describe())

    st.write("""En este resumen estadístico se puede observar que la media es de aproximadamente 24.002 grados Celsius. Esto significa que, en promedio, las temperaturas registradas rondan los 24 grados Celsius. Al analizar la desviación estándar que es de aproximadamente 4.151633 grados Celsius, deducimos que las lecturas individuales de temperatura pueden desviarse en promedio alrededor de 4 grados Celsius de la media. Además, la temperatura mínima registrada es de 6.0 grados Celsius y la máxima es de 37.34 grados. En el primer cuartil el 25% más bajo de las lecturas de temperatura son menores o iguales a 20.98 grados. El percentil 50, también conocido como la mediana, es el valor que divide el conjunto de datos ordenado en dos partes iguales. En este caso, el 50% de las lecturas de temperatura son menores o iguales a 23.91 grados. En este caso, el 75% (tercer cuartil) más bajo de las lecturas de temperatura son menores o iguales a 27.37 grados Celsius.""")

    # Graficar las temperaturas
    st.subheader('Gráfico de Temperaturas')
    plt.figure(figsize=(10, 6))
    plt.plot(df_temperature['Fecha'], df_temperature['Temperatura'], marker="o", linestyle="-", color="b")
    plt.xlabel("Fecha (MM-DD)")
    plt.ylabel("Temperatura (°C)")
    plt.xticks(rotation=45) 
    st.pyplot()

    st.write("""En este gráfico podemos observar la distribución de las temperaturas normales y anómalas. Se puede observar que la mayoría de las temperaturas normales se encuentran en un rango de 20 a 28 grados Celsius, mientras que las temperaturas anómalas se encuentran en un rango de 5 a 40 grados Celsius. """)
    
    # Identificar y contar anomalías
    anomalous_readings = df_temperature[(df_temperature["Temperatura"] < 10.0) | (df_temperature["Temperatura"] > 35.0)]
    num_anomalous_readings = len(anomalous_readings)

    st.subheader('Lecturas Anómalas')
    if num_anomalous_readings > 0:
        st.write("Número de lecturas anómalas:", num_anomalous_readings)
        st.write(anomalous_readings)
    else:
        st.write("No se encontraron lecturas anómalas.")

    st.write("""En esta parte del código, ubicamos las lecturas anómalas, donde nos ubican 5 lecturas anómalas, que son las siguientes: 7.16, 37.34, 6.06, 35.81, 6.52. Estas lecturas anómalas son las que se encuentran fuera del rango de 18 a 30 grados Celsius.""")
    
    # Gráfico de caja y bigotes
    st.subheader('Distribución de las Lecturas de Temperatura')
    plt.figure(figsize=(8, 6))
    plt.boxplot(df_temperature["Temperatura"], vert=False)
    plt.xlabel("Temperatura (°C)")
    st.pyplot()

    st.write("""Con el gráfico de caja logramos ver la distribución de las temperaturas normales y anómalas. Se puede observar que la mayoría de las temperaturas normales se encuentran en un rango de 20 a 28 grados Celsius, indicándonos las media de 24 grados, mientras que logramos observar los outliers que son las temperaturas anómalas que la mayoría se encuentran en un rango de 5 a 8 grados.""")

    # Histograma de las temperaturas
    st.subheader('Histograma de las Lecturas de Temperatura')
    plt.figure(figsize=(8, 6))
    plt.hist(df_temperature["Temperatura"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Frecuencia")
    st.pyplot()

    st.write("""Con el histogrma logramos ver la distribución de las temperaturas normales y anómalas. Se puede observar que alrededor de los 22 grados Celsius es donde se encuentran la mayoría de las temperaturas normales.""")
    
    # Gráfico de barras para comparar temperaturas normales y anómalas por mes
    df_temperature["Month"] = df_temperature["Fecha"].apply(lambda x: x.split("-")[0])
    normal_temperatures = df_temperature[df_temperature["Temperatura"] >= 10.0]
    anomalous_temperatures = df_temperature[df_temperature["Temperatura"] < 10.0]

    monthly_normal_counts = normal_temperatures.groupby("Month").size()
    monthly_anomalous_counts = anomalous_temperatures.groupby("Month").size()

    months = df_temperature["Month"].unique()
    normal_counts = [monthly_normal_counts[month] if month in monthly_normal_counts else 0 for month in months]
    anomalous_counts = [monthly_anomalous_counts[month] if month in monthly_anomalous_counts else 0 for month in months]

    plt.figure(figsize=(10, 6))
    plt.bar(months, normal_counts, color="b", label="Temperaturas Normales")
    plt.bar(months, anomalous_counts, color="r", bottom=normal_counts, label="Temperaturas Anómalas")
    plt.xlabel("Mes")
    plt.ylabel("Cantidad de lecturas")
    plt.title("Comparación de temperaturas normales y anómalas por mes")
    plt.legend()
    st.pyplot()

    st.write("""Este gráfico nos muestra la comparación de las temperaturas normales y anómalas, de acuerdo a la cantidad de lecturas de cada en cada mes. Se puede observar que en el mes de enero,marzo,abril,mayo,junio,agosto,septiembre,octubre y diciembre se registraron todas las temperaturas normales que anómalas, mientras que en el mes de febrero,julio y noviembre se registraron algunas temperaturas anómalas.""")

    # Detección de anomalías con Isolation Forest
    iso_forest = IsolationForest(contamination=0.05)
    anomalies = iso_forest.fit_predict(df_temperature[['Temperatura']])
    df_temperature['Anomaly'] = anomalies == -1

    # Gráfico de temperaturas y anomalías
    st.subheader('Temperaturas Diarias con Anomalía Detectada')
    plt.figure(figsize=(15, 6))
    plt.plot(df_temperature['Fecha'], df_temperature['Temperatura'], label='Temperaturas')
    plt.scatter(df_temperature.loc[df_temperature['Anomaly'], 'Fecha'], df_temperature.loc[df_temperature['Anomaly'], 'Temperatura'],
                color='red', label='Anomalía', marker='x', s=100)
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')
    plt.title('Temperaturas Diarias con Anomalía Detectada')
    plt.legend()
    st.pyplot()

    st.header("Análisis de los resultados")
    st.write("""La gráfica resultante nos muestra las transacciones a lo largo del año con marcas rojas en los puntos donde las transacciones fueron anormalmente altas y bajas  según la detección realizada por Isolation Forest. Estas marcas nos ayudan a visualizar rápidamente cuándo y en que temperatura ocurrieron estos eventos fuera de lo común y que podrían requerir investigación adicional, ya que podrían ser indicativos de problemas en el sensor o en el sistema de monitoreo de temperatura.""")
    st.write("""En conclusión, la detección de anomalías en un conjunto de datos de temperatura implica identificar lecturas que se desvían significativamente de la tendencia central y de la distribución general de las lecturas. Se pueden utilizar métodos estadísticos, como la desviación estándar o los percentiles, para identificar estas anomalías y tomar medidas adecuadas, y nos puedan servir como indicadores tempranos de problemas potenciales en el proceso de fabricación o en los equipos, permitiendo una intervención rápida para minimizar el impacto en la calidad del producto y en la eficiencia operativa. Además, los métodos de detección de anomalías basados en el aprendizaje automático, como Isolation Forest, pueden ser útiles para identificar patrones inusuales en los datos y detectar anomalías de manera eficiente.""")


def tema_tye():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    # Generar serie temporal simulada
    np.random.seed(0)
    t = np.arange(120)
    data = 20 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(size=120)
    serie_temporal = pd.Series(data, index=pd.date_range(start='2010-01-01', periods=120, freq='M'))

    # Configurar la aplicación de Streamlit
    st.title('Análisis de Estacionariedad de una Serie Temporal de Manufactura')

    st.header("Análisis de Estacionariedad de una Serie Temporal de manufactura")

    # Mostrar la serie temporal
    st.subheader('Serie Temporal')
    st.line_chart(serie_temporal)

    st.write("""En la primera parte del código tenemos los datos generados que simulan la temperatura media mensual en grados Celsius a lo largo de 10 años con una tendencia y estacionalidad anual.""")
    
    # Generar estadísticas descriptivas de la serie temporal
    st.subheader('Estadísticas Descriptivas')
    st.write(serie_temporal.describe())

    st.write("""
            La serie temporal tiene un total de 120 observaciones. La temperatura media registrada es de 
             aproximadamente 23.1076. La desviación estándar de alrededor de 7.34 señala la variabilidad en las 
             temperaturas, lo que sugiere fluctuaciones en el clima a lo largo del tiempo.

            El valor mínimo registrado es de aproximadamente 9.6692, mientras que el máximo es de aproximadamente 
             36.4560. Esto indica la gama de temperaturas experimentadas en la ciudad durante los 10 años analizados, 
             lo que puede reflejar las estaciones del año o eventos climáticos extremos.

            Los percentiles (25%, 50%, 75%) ofrecen una visión de la distribución de las temperaturas. Por ejemplo, 
             el 25% de las temperaturas están por debajo de 16.77, lo que indica la prevalencia de temperaturas más 
             bajas en la ciudad. La mediana, que está alrededor de 22.85, sugiere que la mitad de las temperaturas 
             están por debajo de este valor, y el 75% de las temperaturas están por debajo de 30.19, lo que muestra 
             que las temperaturas más altas son menos comunes pero aún ocurren.""")
    
    # Visualizar serie temporal con media y varianza móvil
    rolling = serie_temporal.rolling(window=12)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    st.subheader('Media y Varianza Móvil')
    st.line_chart(rolling_mean)
    st.line_chart(rolling_std)

    st.write("""Con ayuda de los dos gráficos anteriores, podemos ver que la serie temporal de manera visual no es estacionaria, ya que la media y la varianza no son constantes a lo largo del tiempo. Por lo tanto, necesitamos transformar la serie temporal en una serie estacionaria antes de poder aplicar algunos métodos de series temporales.""")

    # Diferenciación de la serie temporal
    serie_temporal_diff = serie_temporal.diff().dropna()

    # Mostrar serie temporal diferenciada
    st.subheader('Serie Temporal Diferenciada')
    st.line_chart(serie_temporal_diff)

    # Media y varianza móvil de la serie temporal diferenciada
    rolling_diff = serie_temporal_diff.rolling(window=12)
    rolling_mean_diff = rolling_diff.mean()
    rolling_std_diff = rolling_diff.std()

    st.subheader('Media y Varianza Móvil de la Serie Temporal Diferenciada')
    st.line_chart(rolling_mean_diff)
    st.line_chart(rolling_std_diff)

    # Prueba de Dickey-Fuller aumentada (ADF) para la serie temporal original
    st.subheader('Prueba de Dickey-Fuller Aumentada (ADF)')
    result_adf = adfuller(serie_temporal)
    st.write('ADF Statistic:', result_adf[0])
    st.write('p-value:', result_adf[1])
    if result_adf[1] <= 0.05:
        st.write('La serie temporal es estacionaria')
    else:
        st.write('La serie temporal no es estacionaria')

    # Prueba de Dickey-Fuller aumentada (ADF) para la serie temporal diferenciada
    st.subheader('Prueba de Dickey-Fuller Aumentada (ADF) para la Serie Temporal Diferenciada')
    result_adf_diff = adfuller(serie_temporal_diff)
    st.write('ADF Statistic:', result_adf_diff[0])
    st.write('p-value:', result_adf_diff[1])
    if result_adf_diff[1] <= 0.05:
        st.write('La serie temporal diferenciada es estacionaria')
    else:
        st.write('La serie temporal diferenciada no es estacionaria')


    st.write("""En cambio, en la gráfica que se muestró a continuación junto con los valores de la prueba ADF es de la serie temporal después de aplicar la transformación, y los valores de las estadísticas de la prueba ADF son menores que los valores críticos, ya que el valor de p fue igual a 0.0000, lo que indica que podemos rechazar la hipótesis nula de que la serie temporal no es estacionaria, esto se debe a que los datos no tienen una raíz unitaria. Por lo tanto, podemos concluir que la serie temporal después de aplicar la transformación es estacionaria.""")
    st.write("""En conclusión, realizar pruebas estadísticas formales como lo son la Prueba de Dickey-Fuller Aumentada (ADF) para determinar la estacionariedad de una serie temporal es crítico para garantizar la validez de los modelos y análisis posteriores, así como para mejorar la precisión de las predicciones a futuro. Estas pruebas son fundamentales en el proceso de análisis de series temporales y deben llevarse a cabo para obtener resultados confiables y útiles. """)


def main():
    st.sidebar.title("Menú de Temas")
    tema_seleccionado = st.sidebar.selectbox("Seleccione un tema", ["Estadísticas Descriptivas", "Patrones en Series Temporales", "Anomalías en Series Temporales","Transformaciones y Estacionaridad en Series Temporales"])

    st.sidebar.subheader("Alumna: Olga Yarely Gutiérrez Martínez")
    st.sidebar.subheader("Materia: Analisis de series temporales")
    st.sidebar.subheader("Docente: Dr. Walter Alexander Mata Lopez")
    st.sidebar.subheader("Facultad de Ingenieria Mecanica y Electrica")
    st.sidebar.subheader("Universidad de Colima")
    st.sidebar.subheader("Virnes, 26 de Abril de 2024")
    st.sidebar.subheader("Proyecto de Ejercicios de la 2da parcial en Streamlit")
    
    

    if tema_seleccionado == "Estadísticas Descriptivas":
        tema_estadisticas_descriptivas()
    elif tema_seleccionado == "Patrones en Series Temporales":
        tema_patrones()
    elif tema_seleccionado == "Anomalías en Series Temporales":
        tema_anomalias()
    elif tema_seleccionado == "Transformaciones y Estacionaridad en Series Temporales":
        tema_tye()


if __name__ == "__main__":
    main()
