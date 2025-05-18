import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px
import ast
st.header('Param dist used to get cluster labels :bar_chart:')
st.write(" min_cluster_size: [5,10,25,50,75,100,125,150,175,200,225,250,275,300]")
st.write(" min_samples: [5,10,25,50,75,100,125,150,175,200]")
st.write(" cluster_selection_method:['leaf','eom']")

st. write("If no combo exist it's because it produced a cluster with under 3 labels or over 40 labels")

# Load and prepping data
@st.cache_data # stores in caches so it doesnt have to recompute everytime 
def load_data():

    # feat_data contain all the features of the clusters such as scores, params, and labels 
    # text_data contains all the text 
    # vec_data_3D and vec_data_2D are the TSNE rep of data 

    vec_data_3D = np.load("/ocean/projects/cis240109p/mmarius/PNC_work/NLP/TSNE_3.npy") 
    vec_data_2D = np.load("/ocean/projects/cis240109p/mmarius/PNC_work/NLP/TSNE.npy") # Simulate an expensive operation
    train=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/train.csv')
    test=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/test.csv')
    text_data=pd.concat([train,test]).reset_index()
    text_data=text_data.drop(columns=['index'])

    part_1_init= pd.read_csv('hdbscan_optimization_results_4.csv')
    part_1 = pd.concat([pd.json_normalize(part_1_init['params'].apply(ast.literal_eval)), part_1_init[['labels', 'score','prop_score']]], axis=1)

    part_2_init= pd.read_csv('hdbscan_optimization_results_5.csv')
    part_2 = pd.concat([pd.json_normalize(part_2_init['params'].apply(ast.literal_eval)), part_2_init[['labels', 'score','prop_score']]], axis=1)

    part_3_init= pd.read_csv('hdbscan_optimization_results_6.csv')
    part_3 = pd.concat([pd.json_normalize(part_3_init['params'].apply(ast.literal_eval)), part_3_init[['labels', 'score','prop_score']]], axis=1)

    part_4_init= pd.read_csv('hdbscan_optimization_results_7.csv')
    part_4 = pd.concat([pd.json_normalize(part_4_init['params'].apply(ast.literal_eval)), part_4_init[['labels', 'score','prop_score']]], axis=1) 
    
    part_5_init= pd.read_csv('hdbscan_optimization_results_8.csv')
    part_5 = pd.concat([pd.json_normalize(part_5_init['params'].apply(ast.literal_eval)), part_5_init[['labels', 'score','prop_score']]], axis=1) 
    
    part_6_init= pd.read_csv('hdbscan_optimization_results_10.csv')
    part_6 = pd.concat([pd.json_normalize(part_6_init['params'].apply(ast.literal_eval)), part_6_init[['labels', 'score','prop_score']]], axis=1) 
     
    feat_df = pd.concat([part_1, part_2,part_3,part_4,part_5,part_6], axis=0) 
    
    return vec_data_3D,vec_data_2D,text_data,feat_df # replace part_1 with feat_df when have full code 

v_data_3D,v_data_2D,text_data,feat_df = load_data() # get data



# Sidebar controls
st.sidebar.header("Clustering Parameters HDBSCAN")
# based on the options that the users choose, use this to get labels and scores to plot
min_samples = st.sidebar.select_slider("Number of min samples",options=np.unique(feat_df['min_samples']), key='sample_slider')
min_cluster_size= st.sidebar.select_slider("Number of min cluster size",options=np.unique(feat_df['min_cluster_size']), key='clussize_slider')
cluster_selection_method = st.sidebar.select_slider("leaf or eom",options=np.unique(feat_df['cluster_selection_method']), key='method_slider')


# checking to see if combination even exist choosing labels and props based on selection
labels= feat_df[
    (feat_df['min_samples'] == min_samples) & 
    (feat_df['min_cluster_size'] == min_cluster_size) & 
    (feat_df['cluster_selection_method'] == cluster_selection_method)
]['labels']
# same for scores 
prop_score= feat_df[
    (feat_df['min_samples'] == min_samples) & 
    (feat_df['min_cluster_size'] == min_cluster_size) & 
    (feat_df['cluster_selection_method'] == cluster_selection_method)
]['prop_score']

# checking to see if combination has value
if len(labels)==0:
    st.warning("This combination doesn't have labels")
else:
    labels=ast.literal_eval(labels.iloc[0])

if len(prop_score)==0 or pd.notna(prop_score.iloc[0])==False:
    st.warning("This combination doesn't have scores")
else:
    prop_score=ast.literal_eval(prop_score.iloc[0])



# Making plots 

st.header('Plotting 3D :three:', divider=True)
# making daata frame to plot all info easily
plt_data = pd.DataFrame(v_data_3D, columns= ['t-SNE1','t-SNE2','t-SNE3'])
plt_data['labels']=labels
plt_data['text']=text_data['text']
plt_data['score']=prop_score
plt_data['t-SNE1_2d']=v_data_2D[:,0]
plt_data['t-SNE2_2d']=v_data_2D[:,1]

# Create interactive 3D plot
# color_discrete_sequence=px.colors.qualitative.Dark24
fig=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='labels', color_continuous_scale='viridis'
               , opacity=1,hover_data=['labels','text','score'],
                title=f"HBDSCAN Clustering (min_samp: {min_samples}, min_clus: {min_cluster_size}, type: {cluster_selection_method})")

fig.update_traces(marker=dict(size=3))
fig.update_layout(
    xaxis_title="Feature 1",
    yaxis_title="Feature 2"
    
)
st.plotly_chart(fig)


#Creating 2D plot 
st.header('Plotting 2D :two:',divider=True)
fig_2=px.scatter(plt_data,x='t-SNE1_2d',y='t-SNE2_2d', color='labels', color_continuous_scale='viridis'
               , opacity=1,hover_data=['labels','text','score'],
                title=f"HBDSCAN Clustering (min_samp: {min_samples}, min_clus: {min_cluster_size}, type: {cluster_selection_method})")

fig_2.update_layout(
    xaxis_title="Feature 1",
    yaxis_title="Feature 2"
    )

st.plotly_chart(fig_2)