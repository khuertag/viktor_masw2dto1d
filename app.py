


from viktor import ViktorController
#from viktor.geometry import Point, Sphere
#from viktor.views import GeometryView, GeometryResult, DataView, DataResult, DataGroup, DataItem
from viktor.parametrization import ViktorParametrization, OptionField, TextField, DownloadButton, AutocompleteField
from viktor.views import PlotlyResult, PlotlyView, PlotlyAndDataResult, PlotlyAndDataView, DataGroup, DataItem
from viktor.parametrization import FileField, NumberField #, ChoiceField
from viktor.result import DownloadResult #, PlotResult



from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

# Parametrization Class
class Parametrization(ViktorParametrization):
    uploaded_file = FileField('Subir archivo XYZ:', file_types=['.xyz', '.txt'], max_size=5_000_000)
    tipo_grafico = OptionField('Seleccionar Tipo de Grafico de XYZ:', options=['grid', 'scale', 'contour'])
    distancia = NumberField('Introducir la distancia de extracción del perfil:', default=30)
    delta_velocidad = NumberField('Introducir el delta del contorno de velocidad:', default=50)
    tipo_extraccion = OptionField('Seleccionar Tipo de Extracción:', options=['default', 'delta', 'rango'])
    parametro_extra = NumberField('Ingresar Parámetro para Extracción:', default=1)

    download_btn = DownloadButton('Download file', method = 'download_file')



# Controller Class
class Controller(ViktorController):
    label = 'Aplicación para Extraer y Visualizar Perfil MASW2D'
    parametrization = Parametrization

    def plot_profile(self, params, **kwargs):
        # Your plotting logic here
        pass

    def download_csv(self, params, **kwargs):
        # Your CSV download logic here
        pass
        return DownloadResult(file_content = '', file_name='my_file.txt')
    

# Add your existing functions here, modifying them to return results rather than using Streamlit's st.pyplot or st.download_button

# Function to plot 2D profile (MASW2D)
def graficar_perfil2D(data, tipo_grafico='grid', delta_velocidad=20):
    plt.figure(figsize=(12, 6))

    x = data['X']
    y = data['Y']
    vs = data['Vs']
    cmap = 'RdYlGn'

    if tipo_grafico == 'grid':
        plt.scatter(x, y, c=vs, cmap=cmap, s=5)
    else:
        xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), vs, (xi, yi), method='linear')

        vmax_with_delta = 10 * ((vs.max() + 2*delta_velocidad)//10)  # Include delta in the max value
        
        if tipo_grafico == 'scale':
            plt.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap=cmap, aspect='auto', vmin=10 * (vs.min()//10), vmax=vmax_with_delta)
        elif tipo_grafico == 'contour':
            levels = np.arange(vs.min(), vmax_with_delta, delta_velocidad)
            plt.contourf(xi, yi, zi, levels=levels, cmap=cmap, extend='max')  # Ensure extending to max value
            plt.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)

    plt.colorbar(label='Vs (Velocidad de ondas de corte, m/s)')
    plt.xlabel('X (Ubicación en línea sísmica)')
    plt.ylabel('Y (Cota topográfica)')
    plt.title('Perfil MASW2D (Mapa de Calor)')
    #plt.pyplot(plt)
    plt.show()



# Modified function to extract the velocity profile at a given distance
def extraer_perfil_X_corregido(distancia, data):
    # Selecting data at the specified distance
    perfil_data = data[data['X'] == distancia][['Y', 'Vs']]
    # Finding the surface elevation (maximum Y value)
    cota_superficie = perfil_data['Y'].max()
    # Calculating depth by subtracting surface elevation and taking absolute value
    perfil_data['Y'] = (perfil_data['Y'] - cota_superficie).abs()
    # Renaming columns to represent depth and velocity
    perfil_data.columns = ['Profundidad', 'Velocidad (m/s)']
    return perfil_data


# Function to plot the velocity profile with a stepped appearance
def graficar_perfil_escalones(perfil_data, distancia):
    plt.figure(figsize=(8, 6))
    plt.step(perfil_data['Velocidad (m/s)'], perfil_data['Profundidad'], where='post', color='red', label=f'Distancia = {distancia} m')
    plt.xlabel('Velocidad de ondas de corte (m/s)')
    plt.ylabel('Profundidad (m)')
    plt.title('Perfil de Velocidades (Apariencia Escalonada)')
    plt.legend()
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)  # Adding grid lines
    plt.gca().invert_yaxis()  # Inverting Y-axis to represent depth
    #plt.pyplot(plt)
    plt.show()

# Function to extract the profile to a CSV file
def extraer_csv(perfil_extraido, tipo_extraccion, profundidad=1, vector_profundidad=None):
    perfil_to_save = perfil_extraido.copy()

    # Handling different extraction types
    if tipo_extraccion == 'delta':
        vector_profundidad = np.arange(profundidad, perfil_extraido['Profundidad'].max(), profundidad)
        interpolator = interp1d(perfil_extraido['Profundidad'], perfil_extraido['Velocidad (m/s)'], kind='linear', fill_value='extrapolate')
        velocities = interpolator(vector_profundidad)
        perfil_to_save = pd.DataFrame({'Profundidad': vector_profundidad, 'Velocidad (m/s)': velocities})
    elif tipo_extraccion == 'rango':
        interpolator = interp1d(perfil_extraido['Profundidad'], perfil_extraido['Velocidad (m/s)'], kind='linear', fill_value='extrapolate')
        velocities = interpolator(vector_profundidad)
        perfil_to_save = pd.DataFrame({'Profundidad': vector_profundidad, 'Velocidad (m/s)': velocities})
    
    # Converting the DataFrame to CSV format
    csv = perfil_to_save.to_csv(index=False)
    return csv
