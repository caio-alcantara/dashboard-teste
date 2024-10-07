import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

@st.cache_resource
def load_anomaly_model():
    return joblib.load('iso_forest_tuning_model.pkl')

@st.cache_data
def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def predict_anomalies(model, df_scaled):
    labels = model.predict(df_scaled)
    return pd.Series(labels).map({1: 0, -1: 1})


st.image("galvao.png", width=350)

st.markdown("<h1 style='text-align: left;'><span style='color: #c9a487;'>Método Galvão</span> para identificação de anomalias e previsão de consumo de gás</h1>", unsafe_allow_html=True)
st.markdown("### Utilize as abas abaixo para conhecer os objetivos da dashboard, analisar anomalias ou fazer predições de consumo.")

tab1, tab2, tab3 = st.tabs(["Apresentação", "Detecção de Anomalias", "Previsão de Consumo"])

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Aba 1: Apresentação
with tab1:
    st.header("Objetivos da Dashboard")
    st.write("""
    Esta dashboard foi desenvolvida com a finalidade de tornar mais fácil a identificação de anomalias no consumo de gás de clientes e a realização de previsões de consumo para clientes individuais.                      
    
    **Objetivos:**
    - **Detecção de Anomalias**: Utilizar o modelo Isolation Forest para identificar comportamentos anômalos nos dados de consumo de gás.
    - **Predição de Consumo**: Utilizar o modelo Holt-Winters para prever o consumo de gás de um cliente individual com base no histórico de consumo.
    
    **Aviso Importante**: Os arquivos CSV que devem ser subidos nessa dashboard devem ser aqueles exportados ao final da execução do notebook do projeto.
    
    Navegue pelas abas acima para explorar as funcionalidades da dashboard.
    """)

    st.header("Sobre o Método Galvão")
    st.write("""
    O Método Galvão é uma técnica de análise de dados que utiliza modelos de Machine Learning para identificar anomalias e prever o consumo de gás de clientes.
    Foi desenvolvido pela equipe Galvão & Associados Gases e Dados, com o objetivo de melhorar a eficiência na gestão de consumo de gás.
    
    A equipe é composta por Caio de Alcantara Santos, Cecília Beatriz Melo Galvão, Pablo de Azevedo, Lucas Cozzolino Tort, Nataly de Souza Cunha, Kethlen Martins da Silva, Mariella Sayumi Mercado Kamezawa.
    """)


# Aba 2: Detecção de Anomalias
with tab2:
    st.header("Análise de Anomalias com Isolation Forest")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("arvore.png", width=200)

    st.write("""
    Esta ferramenta usa o modelo **Isolation Forest** para detectar anomalias nos dados de consumo de gás. 
    Faça o upload do arquivo CSV disponibilizado após executar o notebook do projeto (df_sample_selecionado.csv).
    """)
    
    
    # Carregar o modelo de anomalias
    iso_forest_model = load_anomaly_model()

    uploaded_file = st.file_uploader("Carregar arquivo CSV", type="csv", key="anomalias")

    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)

        df_filtered = df_new.drop(columns=['clientCode', 'clientIndex', 'clientCode_encoded', 'delta_time', 'consumo_horarizado'])

        st.write("Visualização dos primeiros 3 dados carregados:")
        st.dataframe(df_filtered.head(3))

        selected_columns = df_filtered.select_dtypes(include=["number"]).columns

        if len(selected_columns) > 0:
            df_selected = df_filtered[selected_columns]
            df_scaled = scale_data(df_selected)

            df_new['anomaly'] = predict_anomalies(iso_forest_model, df_scaled)

            num_anomalias = df_new[df_new['anomaly'] == 1].shape[0]
            num_normais = df_new[df_new['anomaly'] == 0].shape[0]

            st.markdown(f"<span style='color: red;'>**Número de anomalias detectadas:** {num_anomalias}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: green;'>**Número de amostras normais:** {num_normais}</span>", unsafe_allow_html=True)

            st.subheader("Instalações com Dados Anômalos")

            df_anomalies = df_new[df_new['anomaly'] == 1][['clientCode', 'clientCode_encoded', 'clientIndex', 'delta_time', 'consumo_horarizado', 'anomaly']]

            df_anomalies_unique = df_anomalies.drop_duplicates(subset=['clientCode', 'clientIndex'])
            df_anomalies_unique = df_anomalies_unique.sort_values(by='delta_time', ascending=False)

            st.write("Tabela de instalações que possuem dados anômalos (Em primeiro, dados com tempo muito longo entre medições):")
            st.dataframe(df_anomalies_unique)

            st.subheader("Visualização de Anomalias")
            st.write("Gráfico de contagem de amostras normais vs anomalias:")

            st.bar_chart(df_new['anomaly'].value_counts())

            st.subheader("Scatterplot de Anomalias vs Delta Time")
            st.write("O gráfico abaixo mostra a relação entre as anomalias detectadas e o tempo decorrido desde a última medição.")
            st.write("As amostras normais estão em azul e as anomalias estão em vermelho. As principais anomalias são aquelas em que se passou muito tempo entre uma medição e outra.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='delta_time', y='consumo_horarizado', hue='anomaly', data=df_new, palette={0: 'blue', 1: 'red'}, alpha=0.7, ax=ax
            )
            ax.set_title('Anomalias vs Delta Time')
            ax.set_xlabel('Delta Time (horas)')
            ax.set_ylabel('Variação de Consumo por Hora')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['Normal', 'Anomalia'], title='Classificação')
            ax.grid(True)
            st.pyplot(fig)
            
            st.subheader("Distribuição de Consumo Diário (Normais vs Anômalias)")
            fig_box, ax_box = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='anomaly', y='consumo_horarizado', data=df_new, palette='Set1', ax=ax_box)
            ax_box.set_title("Boxplot de Consumo Diário - Normais vs Anômalias")
            ax_box.set_xlabel("Classificação")
            ax_box.set_ylabel("Consumo Horarizado (m³)")
            st.pyplot(fig_box)

            st.subheader("Mapa de Correlação das Variáveis")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            corr_matrix = df_new[selected_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
            ax_corr.set_title("Mapa de Correlação das Variáveis")
            st.pyplot(fig_corr)
            
        else:
            st.warning("Não foram encontradas colunas numéricas no arquivo carregado.")
    else:
        st.warning("Por favor, carregue um arquivo CSV.")


# Aba 3: Predição de Consumo de um Cliente (Somente Visual)
with tab3:
    st.header("Predição de Consumo de Gás para um Cliente utilizando Holt-Winters")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("grafico.png", width=200)
    
    st.write("""
    Esta ferramenta usa o modelo **Holt-Winters** para prever o consumo de um cliente presente nos dados sintéticos.
    O modelo Holt-Winters utiliza equações estatísticas para realizar a previsão de séries temporais, levando em consideração
    a sazonalidade e tendência dos dados. 
    Aqui, você poderá escolher uma instalação única (clientCode e clientIndex) e visualizar a previsão de consumo de gás para essa instalação.
    **Para este exemplo, não será necessário realizar upload de dados. Você poderá utilizar os dados 
    de amostra que disponibilizamos já internamente na dashboard.**
    """)
    
    #uploaded_file = st.file_uploader("Carregar arquivo CSV", type="csv", key="previsao")
    uploaded_file = True
    if uploaded_file is not None:
        #df_50_instalacoes = pd.read_csv(uploaded_file)
        df_50_instalacoes = pd.read_csv('df_50_instalacoes.csv')
        
        label_encoder = LabelEncoder()
        df_50_instalacoes['clientCode_encoded'] = label_encoder.fit_transform(df_50_instalacoes['clientCode'])
        
        cols = ['clientCode_encoded'] + [col for col in df_50_instalacoes.columns if col != 'clientCode_encoded']
        df_50_instalacoes = df_50_instalacoes[cols]
        
        st.write("Visualização dos primeiros 3 dados carregados:")
        st.dataframe(df_50_instalacoes.head(3))
        
        client_code = st.selectbox("Selecione o clientCode:", df_50_instalacoes['clientCode_encoded'].unique())

        ## Identificar os clientIndices associados ao client
        client_indices = df_50_instalacoes[df_50_instalacoes['clientCode_encoded'] == client_code]['clientIndex'].unique()

        ## Selecionar um clientIndex baseado no clientCode
        client_index = st.selectbox("Selecione o clientIndex:", client_indices)
        
        instalacao_df = df_50_instalacoes[(df_50_instalacoes['clientCode_encoded'] == client_code) & (df_50_instalacoes['clientIndex'] == client_index)].copy()
        
        instalacao_mensal_df = instalacao_df.groupby('ano_mes')['consumo_dia'].sum().reset_index()

        instalacao_mensal_df.set_index('ano_mes', inplace=True)

        fig_consumo_real = px.line(
            instalacao_mensal_df.reset_index(), 
            x='ano_mes', 
            y='consumo_dia', 
            title='Consumo Mensal Real (m³) por Ano/Mês',
            labels={'ano_mes': 'Ano/Mês', 'consumo_dia': 'Consumo Diário (m³)'},
        )

        st.plotly_chart(fig_consumo_real)
        
        # Gráfico de Média Móvel de 7 e 30 dias
        fig_media_movel = px.line(
            instalacao_df,
            x='data_hora',
            y=['media_movel_7_dias', 'media_movel_30_dias'],
            title='Média Móvel de 7 e 30 dias',
            labels={'data_hora': 'Data', 'value': 'Consumo Médio (m³)'},
        )

        st.plotly_chart(fig_media_movel)
                
        # Gráfico de Variação Percentual
        fig_mudanca_percentual = px.line(
            instalacao_df,
            x='data_hora',
            y='mudanca_percentual',
            title='Mudança Percentual no Consumo Diário',
            labels={'data_hora': 'Data', 'mudanca_percentual': 'Mudança Percentual (%)'},
        )

        st.plotly_chart(fig_mudanca_percentual)
        
        # Gráfico de Barras da Média de Consumo Mensal por Estação do Ano
        media_consumo_estacao = instalacao_df.groupby('estacao')['consumo_dia'].mean().reset_index()

        media_consumo_estacao = media_consumo_estacao.sort_values(by='consumo_dia')
        max_consumo_estacao = media_consumo_estacao.loc[media_consumo_estacao['consumo_dia'].idxmax()]

        # Gráfico de Barras da Média de Consumo Diário por Estação do Ano
        fig_media_estacao = px.bar(
            media_consumo_estacao,
            x='estacao',
            y='consumo_dia',
            title='Média de Consumo Diário Para Cada Estação do Ano (Maior Consumo em Vermelho)',
            labels={'estacao': 'Estação do Ano', 'consumo_dia': 'Média de Consumo Diário (m³)'},
        )

        # Destacar a barra com maior consumo em vermelho
        fig_media_estacao.update_traces(marker_color=['red' if estacao == max_consumo_estacao['estacao'] else 'blue' for estacao in media_consumo_estacao['estacao']])

        st.plotly_chart(fig_media_estacao)
        
        # Histograma do Consumo Diário
        fig_hist_consumo = px.histogram(
            instalacao_df,
            x='consumo_dia',
            title='Distribuição do Consumo Diário (m³)',
            labels={'consumo_dia': 'Consumo Diário (m³)'},
            nbins=100
        )

        st.plotly_chart(fig_hist_consumo)

        ##### Previsão de Consumo com Holt-Winters #####
        st.header("Previsão de Consumo com Holt-Winters")
        st.write('Aguarde enquanto o modelo Holt-Winters é ajustado e a previsão é feita...')

        # Parâmetros a serem testados
        trend_options = ['add', 'mul', None]
        seasonal_options = ['add', 'mul', None]
        seasonal_periods = [i for i in range(2, 31)]  
        damped_trend_options = [True, False]

        
        # Variáveis para armazenar os melhores resultados
        best_mae = np.inf
        best_mse = np.inf
        best_mape = np.inf
        best_params = {}
        
        train_size = int(len(instalacao_mensal_df) * 0.8)
        train, test = instalacao_mensal_df.iloc[:train_size], instalacao_mensal_df.iloc[train_size:]
        
        
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                for period in seasonal_periods:
                    for damped in damped_trend_options:
                        try:
                            ## Apenas testar damped_trend se houver uma tendência
                            if trend is None and damped:
                                continue
                            
                            ## Definir e ajustar o modelo Holt-Winters com os dados de treino
                            modelo_treino = ExponentialSmoothing(
                                train['consumo_dia'], 
                                trend=trend, 
                                seasonal=seasonal,
                                seasonal_periods=period,
                                damped_trend=damped
                            )

                            ajuste_treino = modelo_treino.fit()

                            ## Fazer a previsão para o número de meses no conjunto de teste
                            previsao_teste = ajuste_treino.forecast(len(test))

                            ## Calcular as métricas
                            mae = mean_absolute_error(test['consumo_dia'], previsao_teste)
                            mse = mean_squared_error(test['consumo_dia'], previsao_teste)
                            mape = mean_absolute_percentage_error(test['consumo_dia'], previsao_teste)

                            ## Comparar o erro e armazenar o melhor modelo (baseado no MAE)
                            if mae < best_mae:
                                best_mae = mae
                                best_mse = mse
                                best_mape = mape
                                best_params = {
                                    'trend': trend,
                                    'seasonal': seasonal,
                                    'seasonal_periods': period,
                                    'damped_trend': damped
                                }

                        except Exception as e:
                            print(f"Erro com a configuração {trend}, {seasonal}, {period}, damped={damped}: {e}")
    
    
    
        modelo_treino = ExponentialSmoothing(
            instalacao_mensal_df['consumo_dia'], 
            trend=best_params['trend'],      
            seasonal=best_params['seasonal'],   
            seasonal_periods=best_params['seasonal_periods'],
            damped_trend=best_params['damped_trend']  
        )

        ajuste_treino = modelo_treino.fit()

        ## Visualização da previsão
        st.subheader("Previsão de Consumo Mensal")
        
        months_to_predict = st.selectbox("Selecione o número de meses para prever:", range(1, 25))

        ## Prever o consumo para o número de meses selecionado
        previsao_mensal = ajuste_treino.forecast(months_to_predict)

        if months_to_predict == 1:
            st.write(f"Previsão de consumo para essa instalação para o próximo mês:")
        else:
            st.write(f"Previsão de consumo para essa instalação para os próximos {months_to_predict} meses:")
        
        previsao_df = pd.DataFrame({
            'Ano/Mês': previsao_mensal.index,
            'Previsão (m³)': previsao_mensal.values
        })
        
        previsao_df['Ano/Mês'] = previsao_df['Ano/Mês'].dt.strftime('%m/%Y')

        st.write("Tabela de Previsão de Consumo:")
        st.dataframe(previsao_df)
        # Calcular o somatório das previsões
        soma_previsoes = previsao_mensal.sum()
        st.write(f"Somatório das previsões de consumo para o(s) próximo(s) {months_to_predict} mes(es): {soma_previsoes:.2f} m³")

        df_grafico = pd.concat([
            instalacao_mensal_df.reset_index(),  # Dados reais
            pd.DataFrame({'ano_mes': previsao_mensal.index, 'consumo_dia': previsao_mensal.values})  # Dados previstos
        ])

        df_grafico['tipo'] = ['Real' if i < len(instalacao_mensal_df) else 'Previsão' for i in range(len(df_grafico))]

        fig_previsao = px.line(
            df_grafico, 
            x='ano_mes', 
            y='consumo_dia', 
            color='tipo', 
            title=f'Previsão de Consumo para os próximos {months_to_predict} meses' if months_to_predict > 1 else 'Previsão de Consumo para o próximo mês',
            labels={'ano_mes': 'Ano/Mês', 'consumo_dia': 'Consumo (m³)'},
        )

        ## Real (vermelho) e previsão (verde)
        fig_previsao.update_traces(
            line=dict(color='red'),  ## Para dados reais
            selector=dict(name='Real')
        )

        fig_previsao.update_traces(
            line=dict(color='green'),  #R Para dados de previsão
            selector=dict(name='Previsão'),
        )
        
        if months_to_predict != 1:
            st.plotly_chart(fig_previsao)
                        




        