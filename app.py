


from viktor import ViktorController, progress_message
#from viktor.geometry import Point, Sphere
#from viktor.views import GeometryView, GeometryResult, DataView, DataResult, DataGroup, DataItem
from viktor.parametrization import ViktorParametrization, OptionField, TextField, DownloadButton, SetParamsButton, AutocompleteField
from viktor.views import PlotlyResult, PlotlyView, PlotlyAndDataResult, PlotlyAndDataView, DataGroup, DataItem
from viktor.parametrization import FileField, NumberField, ActionButton, HiddenField #, ChoiceField
from viktor.result import DownloadResult #, PlotResult
from viktor.views import DataGroup, DataItem, DataResult, DataView
from viktor.result import SetParamsResult, SetParametersResult
from viktor import File, UserError

from munch import Munch, unmunchify
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from copy import deepcopy
import re
import json
import logging

file_path = Path(__file__).parent / 'debug_viktor.log'
logging.basicConfig(filename=file_path, level=logging.DEBUG)




# Parametrization Class
class Parametrization(ViktorParametrization):
    uploaded_file = FileField('Subir archivo XYZ:', file_types=['.XYZ','.xyz', '.txt'], max_size=5_000_000)
    tipo_grafico = OptionField('Seleccionar Tipo de Grafico de XYZ:', options=['grid', 'scale', 'contour'], default='grid')
    distancia = NumberField('Introducir la distancia de extracción del perfil:', default=30)
    delta_velocidad = NumberField('Introducir el delta del contorno de velocidad:', default=50)
    tipo_extraccion = OptionField('Seleccionar Tipo de Extracción:', options=['default', 'delta', 'rango'], default = 'delta')
    parametro_extra = TextField('Ingresar Parámetro para Descarga:', default="1")
    extract_button = SetParamsButton('Extraer Perfil', method = 'extraer_perfil')
    download_btn = DownloadButton('Download file', method = 'extraer_csv')
    cargar_datos_btn = SetParamsButton('Cargar Datos', method='cargar_datos')
    valor_numerico = HiddenField("valor numerico")
    valor_str = HiddenField("valor str")
    tabla = HiddenField("valores de tabla")
    data = HiddenField("valores de datos")
    perfil_extraido = HiddenField("valores de datos de perfil")

# Controller Class
class Controller(ViktorController):
    label = 'Aplicación para Extraer y Visualizar Perfil MASW2D'
    parametrization = Parametrization


    #@staticmethod
    def cargar_datos(self, params, **kwargs) -> SetParamsResult:
        logging.debug('Entrando a la función cargar_datos')
        file_resource = params.uploaded_file  # Esto debería ser un objeto FileResource
        if file_resource is not None:
            file_object = file_resource.file  # Esto debería ser un objeto File
            file_content = file_object.getvalue()  # Obtener el contenido del archivo como una cadena

            # Convertir cada línea en una lista de flotantes
            data_list = []
            for line in file_content.split('\n'):
                line_values = list(map(float, line.strip().split()))
                data_list.append(line_values)
            
            progress_message(f"Lina1 Data {data_list[0]}")
            
            # Convertir la lista de listas en un DataFrame de pandas
            data = pd.DataFrame(data_list, columns=['X', 'Y', 'Vs'])
            data = data.dropna()
            data_dict = data.to_dict('records')
            progress_message(f"Datos encabezados {data.head()}")
            # Convertir el diccionario a una cadena JSON
            data_json = json.dumps(data_dict)
            progress_message(f"Datos en json {data_json}")
            logging.debug('Saliendo de la función cargar_datos')
            # Actualizar los parámetros existentes y añadir el nuevo 'data'
            
            return SetParamsResult({
                'data': data_json  # Añadir el nuevo parámetro 'data' como una cadena JSON
            })
        


    # Function to visualize imported data in a table3
    @DataView("Resumen Datos Importados", duration_guess=1)
    def visualize_imported_data3(self, params, **kwargs):
        data_json = params.get('data', None)  # Obtener 'data' de params si está disponible

        if data_json is not None:
            # Convertir la cadena JSON de nuevo a un diccionario
            data_dict = json.loads(data_json)

            # Convertir el diccionario a un DataFrame de pandas
            data_df = pd.DataFrame.from_dict(data_dict)
            #data_df = data_dict
            masw2d_group = DataGroup(
                DataItem('Número de puntos', len(data_df)),
                DataItem('Valor mínimo de Vs', data_df['Vs'].min()),
                DataItem('Valor máximo de Vs', data_df['Vs'].max())
            )
            return DataResult(masw2d_group)
        else:
            return DataResult(DataGroup(DataItem('Mensaje', 'No se ha subido ningún archivo')))
                
    # Function to plot 2D profile (MASW2D)
    @PlotlyView("Graficar Perfil 2D", duration_guess=1)
    def graficar_perfil2D(self, params, **kwargs):
        try:
            data_json = params.get('data', None)  # Obtener 'data' de params si está disponible
            if data_json is not None:
                data_dict = json.loads(data_json)
                # Convertir el diccionario de nuevo a DataFrame para el procesamiento
                data = pd.DataFrame.from_dict(data_dict)
                #data = data_dict
                tipo_grafico = params.tipo_grafico
                delta_velocidad = params.delta_velocidad

                #data = pd.read_csv(uploaded_file, sep=' +', engine='python', names=['X', 'Y', 'Vs'])
                #params['data'] = data

                x = data['X']
                y = data['Y']
                vs = data['Vs']

                fig = go.Figure()

                if tipo_grafico == 'grid':
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=vs, colorscale='RdYlGn', size=5)))
                else:
                    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    zi = griddata((x, y), vs, (xi, yi), method='linear')

                    vmax_with_delta = 10 * ((vs.max() + 2*delta_velocidad)//10)  # Include delta in the max value

                    if tipo_grafico == 'scale':
                        fig.add_trace(go.Contour(z=zi, x=xi[0], y=yi[:, 0], colorscale='RdYlGn', zmin=10 * (vs.min()//10), zmax=vmax_with_delta))
                    elif tipo_grafico == 'contour':
                        levels = np.arange(vs.min(), vmax_with_delta, delta_velocidad)
                        fig.add_trace(go.Contour(z=zi, x=xi[0], y=yi[:, 0], colorscale='RdYlGn', contours=dict(start=vs.min(), end=vmax_with_delta, size=delta_velocidad)))


                fig.update_layout(
                    title=f'Perfil MASW2D ({tipo_grafico})',
                    xaxis_title='X (Ubicación en línea sísmica)',
                    yaxis_title='Y (Cota topográfica)',
                    coloraxis_colorbar=dict(title='Vs (Velocidad de ondas de corte, m/s)')
                )

                if fig is not None:
                    return PlotlyResult(fig.to_json())
                else:
                    return PlotlyResult({"error": "La figura está vacía"})
            else:
                return PlotlyResult({"error": "No se ha subido ningún archivo o los datos no están disponibles"})
        except Exception as e:
            return PlotlyResult({"error": f"Se produjo un error: {str(e)}"})
    
    
    def extraer_perfil(self, params, **kwargs) -> SetParamsResult:
        data_json = params.get('data', None)  # Obtener 'data' de params si está disponible
        distancia = params.distancia

        if data_json is not None:
            data_dict = json.loads(data_json)
            # Convertir el diccionario de nuevo a DataFrame para el procesamiento
            data = pd.DataFrame.from_dict(data_dict)
            data = data.dropna()
            perfil_data = extraer_perfil_X_corregido(distancia, data)
            perfil_dict = perfil_data.to_dict('records')
            progress_message(f"Previo Perfil {perfil_data.head()}")
            # Convertir el diccionario a una cadena JSON
            perfil_json = json.dumps(perfil_dict)
            #progress_message(f"Perfil json {perfil_json}")
            return SetParamsResult({
                'perfil_extraido': perfil_json,  # valor numerico para agregar a params
            })
    
    # Function to visualize extracted profile in a table
    @DataView("Ver Perfil Extraído", duration_guess=1)
    def visualize_extracted_profile(self, params, **kwargs):
        perfil_json = params.get('perfil_extraido', None)  # Assuming 'perfil_extraido' is stored in params

        if perfil_json is not None:
            perfil_dict = json.loads(perfil_json)
            # Convertir el diccionario de nuevo a DataFrame para el procesamiento
            perfil_extraido = pd.DataFrame.from_dict(perfil_dict)
            perfil_extraido = perfil_extraido.dropna()
            progress_message(f"Recuperado Perfil {perfil_extraido.head()}")
            perfil_group = DataGroup(
                DataItem('Número de puntos', len(perfil_extraido)),
                DataItem('Profundidad mínima', perfil_extraido['Profundidad'].min()),
                DataItem('Profundidad máxima', perfil_extraido['Profundidad'].max()),
                DataItem('Velocidad mínima', perfil_extraido['Velocidad (m/s)'].min()),
                DataItem('Velocidad máxima', perfil_extraido['Velocidad (m/s)'].max())
            )

            return DataResult(perfil_group)
        else:
            return DataResult(DataGroup(DataItem('Mensaje', 'No se ha extraído ningún perfil')))
		
	

    # Function to plot the velocity profile with a stepped appearance
    @PlotlyView("Graficar Perfil Escalonado", duration_guess=1)
    def graficar_perfil_escalones(self, params, **kwargs):
        perfil_json = params.get('perfil_extraido', None)  # Assuming 'perfil_extraido' is stored in params
        distancia = params.distancia
        if perfil_json is not None:
            perfil_dict = json.loads(perfil_json)
                # Convertir el diccionario de nuevo a DataFrame para el procesamiento
            perfil_data = pd.DataFrame.from_dict(perfil_dict)
            perfil_data = perfil_data.dropna()
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=perfil_data['Velocidad (m/s)'], y=perfil_data['Profundidad'], mode='lines+markers', line=dict(color='red')))

            fig.update_layout(
                title='Perfil de Velocidades (Apariencia Escalonada)',
                xaxis_title='Velocidad de ondas de corte (m/s)',
                yaxis_title='Profundidad (m)',
                xaxis=dict(gridcolor='gray'),  # Establecer el color de la cuadrícula del eje X
                yaxis=dict(gridcolor='gray', autorange='reversed'),   # Establecer el color de la cuadrícula del eje Y
                plot_bgcolor='rgba(0,0,0,0)'  # Esto establecerá un fondo transparente
            )

            return PlotlyResult(fig.to_json())
        else:
            return PlotlyResult({"error": "No se ha subido ningún archivo o los datos no están disponibles"})


    # Function to extract the profile to a CSV file
    def extraer_csv(self, params, **kwargs):
        
        #perfil_extraido, tipo_extraccion, profundidad=1, vector_profundidad=None
        perfil_json = params.get('perfil_extraido', None)
        perfil_dict = json.loads(perfil_json)
        # Convertir el diccionario de nuevo a DataFrame para el procesamiento
        perfil_data = pd.DataFrame.from_dict(perfil_dict)
        perfil_extraido = perfil_data.dropna()
        tipo_extraccion = params.tipo_extraccion
        parametro_extra_str = params.parametro_extra
        
        if parametro_extra_str: # Convert only if not empty
            if tipo_extraccion == 'delta':
                parametro_extra = float(parametro_extra_str)
            elif tipo_extraccion == 'rango':
                parametro_extra = [float(x) for x in parametro_extra_str.strip('[]').split(',')]

        perfil_to_save = perfil_extraido.copy()

        # Handling different extraction types
        if tipo_extraccion == 'delta':
            profundidad = parametro_extra
            vector_profundidad = np.arange(profundidad, perfil_extraido['Profundidad'].max(), profundidad)
            interpolator = interp1d(perfil_extraido['Profundidad'], perfil_extraido['Velocidad (m/s)'], kind='linear', fill_value='extrapolate')
            velocities = interpolator(vector_profundidad)
            perfil_to_save = pd.DataFrame({'Profundidad': vector_profundidad, 'Velocidad (m/s)': velocities})
        elif tipo_extraccion == 'rango':
            vector_profundidad = parametro_extra
            interpolator = interp1d(perfil_extraido['Profundidad'], perfil_extraido['Velocidad (m/s)'], kind='linear', fill_value='extrapolate')
            velocities = interpolator(vector_profundidad)
            perfil_to_save = pd.DataFrame({'Profundidad': vector_profundidad, 'Velocidad (m/s)': velocities})
        
        # Converting the DataFrame to CSV format
        csv = perfil_to_save.to_csv(index=False)
        return DownloadResult(file_content=csv.encode('utf-8'), file_name='perfil_extraido.csv')





# FUNCION QUE NO INTERACTUA CON LA INTERFAZ GRAFICA DE VIKTOR

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
