import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# 侧边栏设置
st.sidebar.image("hospital_logo2.png", caption="", width=300)
lang = st.sidebar.selectbox('Choose language', ['中文', 'English'])

# 页眉设置
st.image("hospital_logo.png", caption="")
if lang == '中文':  
    st.header("基于机器学习、利用常规实验室检测指标构建的用于结直肠癌诊断的数字生物标志物")
else:  
    st.header("Machine learning-based digital biomarkers using routine laboratory parameters for colorectal cancer diagnosis")

# 输入字段
input_params = [
    ('SG', 'a', 0.000, 2.000, 1.015),
	('CA19-9 u/mL', 'b', 0.00, 100.00, 33.85),
    ('CEA ng/mL', 'c', 0.00, 100.00, 3.85),
    ('Age', 'd', 0, 200, 40),	
    ('ALB g/L', 'e', 0.0, 200.0, 50.0),	
    ('CYFRA21-1 u/mL', 'f', 0.00, 100.00, 2.20),	
    ('HDL-C mmol/L', 'g', 0.0, 100.0, 0.5),
    ('CA72－4 u/mL', 'h', 0.00, 100.00, 10.80)
]

inputs = {}
for param in input_params:
    inputs[param[1]] = st.number_input(
        param[0], 
        min_value=param[2], 
        max_value=param[3], 
        value=param[4]
    )

# 加载模型
#model_dir = r'D:\model\cy3y'
models = {
    'XGBoost': joblib.load(f'xgb_final.pkl')
}

if st.button("Submit"):
    # 创建输入DataFrame
    X = pd.DataFrame([[
        inputs['a'], inputs['b'], inputs['c'], inputs['d'], inputs['e'],
        inputs['f'], inputs['g'], inputs['h']
    ]], columns=[
        "SG", "CA19-9", "CEA", "Age", "ALB", "CYFRA21-1", "HDL-C", "CA72-4"
    ])

    # 进行预测
    for model_name, model in models.items():
        try:
            prob = model.predict_proba(X)[0][1] * 100
            result = f"{prob:.2f}%"
        except Exception as e:
            result = "无法预测（条件不匹配）" if lang == '中文' else "Cannot predict (conditions not match)"
        
        if lang == '中文':
            st.text(f"{model_name} 模型得出的患结直肠癌概率是: {result}")
        else:
            st.text(f"{model_name} model probability for colorectal cancer: {result}")

# 页脚
footer_zh = '<div style="font-size: 12px; text-align: right;">重庆医科大学附属第三医院</div>'
footer_en = '<div style="font-size: 12px; text-align: right;">Chongqing Medical University Affiliated Third Hospital</div>'
st.markdown(footer_zh, unsafe_allow_html=True)
st.markdown(footer_en, unsafe_allow_html=True)
