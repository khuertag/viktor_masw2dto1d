


from viktor import ViktorController
#from viktor.geometry import Point, Sphere
#from viktor.views import GeometryView, GeometryResult, DataView, DataResult, DataGroup, DataItem
from viktor.parametrization import ViktorParametrization, OptionField, TextField, DownloadButton, SetParamsButton, AutocompleteField
from viktor.views import PlotlyResult, PlotlyView, PlotlyAndDataResult, PlotlyAndDataView, DataGroup, DataItem
from viktor.parametrization import FileField, NumberField, ActionButton #, ChoiceField
from viktor.result import DownloadResult #, PlotResult
from viktor.views import DataGroup, DataItem, DataResult, DataView
from viktor.result import SetParamsResult

from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata


# Parametrization Class
class Parametrization(ViktorParametrization):
    uploaded_file = FileField('Subir archivo XYZ:', file_types=['.XYZ','.xyz', '.txt'], max_size=5_000_000)
    tipo_grafico = OptionField('Seleccionar Tipo de Grafico de XYZ:', options=['grid', 'scale', 'contour'], default='grid')
    distancia = NumberField('Introducir la distancia de extracción del perfil:', default=30)
    delta_velocidad = NumberField('Introducir el delta del contorno de velocidad:', default=50)
    tipo_extraccion = OptionField('Seleccionar Tipo de Extracción:', options=['default', 'delta', 'rango'], default = 'delta')
    parametro_extra = TextField('Ingresar Parámetro para Extracción:', default="1")
    extract_button = ActionButton('Extraer Perfil', method = 'extraer_perfil_X_corregido')
    #plot_button = ActionButton('Graficar Perfil MASW2D', method = 'graficar_perfil2D')
    #plot_button_extraido = ActionButton('Graficar Perfil Extraído', method = 'graficar_perfil_escalones')
    download_btn = DownloadButton('Download file', method = 'extraer_csv')
    #visualize_imported_data_button = ActionButton('Visualizar Datos Importados', method='visualize_imported_data')
    #visualize_extracted_profile_button = ActionButton('Visualizar Perfil Extraído', method='visualize_extracted_profile')
    cargar_datos_btn = SetParamsButton('Cargar Datos', method='cargar_datos')

# Controller Class
class Controller(ViktorController):
    label = 'Aplicación para Extraer y Visualizar Perfil MASW2D'
    parametrization = Parametrization

    # Function to visualize imported data in a table
    @DataView("Visualizar Datos Importados", duration_guess=1)
    def visualize_imported_data(self, params, **kwargs):
        data = params.get('data', None)  # Obtener 'data' de params si está disponible

        if data is None:
            file_resource = params.uploaded_file  # Esto debería ser un objeto FileResource
            if file_resource is not None:
                file_object = file_resource.file  # Esto debería ser un objeto File
                file_content = file_object.getvalue()  # Obtener el contenido del archivo como una cadena

                # Convertir cada línea en una lista de flotantes
                data_list = []
                for line in file_content.split('\n'):
                    line_values = list(map(float, line.strip().split()))
                    data_list.append(line_values)

                # Convertir la lista de listas en un DataFrame de pandas
                data = pd.DataFrame(data_list, columns=['X', 'Y', 'Vs'])

                # Almacenar 'data' en params para su uso posterior
                params['data'] = data

        if data is not None:
            masw2d_group = DataGroup(
                DataItem('Número de puntos', len(data)),
                DataItem('Valor mínimo de Vs', data['Vs'].min()),
                DataItem('Valor máximo de Vs', data['Vs'].max()),
            )
            return DataResult(masw2d_group)
        else:
            return DataResult(DataGroup(DataItem('Mensaje', 'No se ha subido ningún archivo')))
    
    def cargar_datos(self, params, **kwargs):
        file_resource = params.uploaded_file  # Esto debería ser un objeto FileResource
        if file_resource is not None:
            file_object = file_resource.file  # Esto debería ser un objeto File
            file_content = file_object.getvalue()  # Obtener el contenido del archivo como una cadena

            # Convertir cada línea en una lista de flotantes
            data_list = []
            for line in file_content.split('\n'):
                line_values = list(map(float, line.strip().split()))
                data_list.append(line_values)

            # Convertir la lista de listas en un DataFrame de pandas
            data = pd.DataFrame(data_list, columns=['X', 'Y', 'Vs'])

            # Actualizar los parámetros existentes y añadir el nuevo 'data'
            return SetParamsResult({
                "uploaded_file": None,  # Limpiar el archivo cargado si lo deseas
                "tipo_grafico": params.tipo_grafico,  # Mantener el valor existente
                "distancia": params.distancia,  # Mantener el valor existente
                "delta_velocidad": params.delta_velocidad,  # Mantener el valor existente
                "tipo_extraccion": params.tipo_extraccion,  # Mantener el valor existente
                "parametro_extra": params.parametro_extra,  # Mantener el valor existente
                "data": data  # Añadir el nuevo parámetro 'data'
            })



    # Function to visualize imported data in a table
    @DataView("Visualizar Datos Importados 2", duration_guess=1)
    def visualize_imported_data2(self, params, **kwargs):
        data_dict = params.get('data', None)  # Obtener 'data' de params si está disponible

        if data_dict is None:
            file_resource = params.uploaded_file  # Esto debería ser un objeto FileResource
            if file_resource is not None:
                file_object = file_resource.file  # Esto debería ser un objeto File
                file_content = file_object.getvalue()  # Obtener el contenido del archivo como una cadena

                # Convertir cada línea en una lista de flotantes
                data_list = []
                for line in file_content.split('\n'):
                    line_values = list(map(float, line.strip().split()))
                    data_list.append(line_values)

                # Convertir la lista de listas en un DataFrame de pandas
                data_df = pd.DataFrame(data_list, columns=['X', 'Y', 'Vs'])

                # Convertir DataFrame a diccionario y almacenar en params
                params['data'] = data_df#.to_dict('records')

        if data_dict is not None:
            # Convertir el diccionario de nuevo a DataFrame para el procesamiento
            #data_df = pd.DataFrame.from_dict(data_dict)
            data_df = data_dict
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
            data_dict = params.get('data', None)  # Obtener 'data' de params si está disponible

            if data_dict is not None:
                # Convertir el diccionario de nuevo a DataFrame para el procesamiento
                #data = pd.DataFrame.from_dict(data_dict)
                data = data_dict
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
                    xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
                    xi, yi = np.meshgrid(xi, yi)
                    zi = griddata((x, y), vs, (xi, yi), method='linear')

                    vmax_with_delta = 10 * ((vs.max() + 2*delta_velocidad)//10)  # Include delta in the max value

                    if tipo_grafico == 'scale':
                        fig.add_trace(go.Contour(z=zi, x=xi[0], y=yi[:, 0], colorscale='RdYlGn', zmin=10 * (vs.min()//10), zmax=vmax_with_delta))
                    elif tipo_grafico == 'contour':
                        levels = np.arange(vs.min(), vmax_with_delta, delta_velocidad)
                        fig.add_trace(go.Contour(z=zi, x=xi[0], y=yi[:, 0], colorscale='RdYlGn', contours=dict(start=vs.min(), end=vmax_with_delta, size=delta_velocidad)))


                fig.update_layout(
                    title='Perfil MASW2D (Mapa de Calor)',
                    xaxis_title='X (Ubicación en línea sísmica)',
                    yaxis_title='Y (Cota topográfica)',
                    coloraxis_colorbar=dict(title='Vs (Velocidad de ondas de corte, m/s)')
                )

                if fig is not None:
                    return PlotlyResult(fig.to_dict())
                else:
                    return PlotlyResult({"error": "La figura está vacía"})
            else:
                return PlotlyResult({"error": "No se ha subido ningún archivo o los datos no están disponibles"})
        except Exception as e:
            return PlotlyResult({"error": f"Se produjo un error: {str(e)}"})

    # Function to visualize extracted profile in a table
    @DataView("Visualizar Perfil Extraído", duration_guess=1)
    def visualize_extracted_profile(self, params, **kwargs):
        perfil_extraido = params.get('perfil_extraido', None)  # Assuming 'perfil_extraido' is stored in params

        if perfil_extraido is not None:
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
        data_dict = params.get('data', None)  # Obtener 'data' de params si está disponible
        distancia = params.distancia

        if data_dict is not None:
            # Convertir el diccionario de nuevo a DataFrame para el procesamiento
            data = pd.DataFrame.from_dict(data_dict)
            
            perfil_data = extraer_perfil_X_corregido(distancia, data)
            params['perfil_extraido'] = perfil_data

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=perfil_data['Velocidad (m/s)'], y=perfil_data['Profundidad'], mode='lines+markers', line=dict(color='red')))

            fig.update_layout(
                title='Perfil de Velocidades (Apariencia Escalonada)',
                xaxis_title='Velocidad de ondas de corte (m/s)',
                yaxis_title='Profundidad (m)',
                yaxis=dict(autorange='reversed'),
                legend_title=dict(text=f'Distancia = {distancia} m'),
                gridcolor='lightgray'
            )

            return PlotlyResult(fig.to_json())
        else:
            return PlotlyResult({"error": "No se ha subido ningún archivo o los datos no están disponibles"})


    # Function to extract the profile to a CSV file
    def extraer_csv(self, params, **kwargs):
        
        #perfil_extraido, tipo_extraccion, profundidad=1, vector_profundidad=None
        perfil_extraido = params.perfil_extraido
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
    
