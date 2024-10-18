# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 데이터 로드 함수 (캐시 사용)
@st.cache_data
def load_data():
    # 데이터 로드
    df_train = pd.read_csv('data/train.csv')
    return df_train

# 데이터 전처리 함수
def preprocess_data(df):
    # 수치형 데이터만 선택
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 수치형 열의 결측치 처리: 중앙값으로 채우기
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 범주형 변수 인코딩: 각 열을 개별적으로 인코딩
    label_encoder = LabelEncoder()
    
    # 범주형 열을 개별적으로 인코딩 처리
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # 결측치는 최빈값으로 대체
        df[col] = label_encoder.fit_transform(df[col])

    # 수치형 데이터 스케일링
    scaler = StandardScaler()
    num_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# 메인 대시보드 함수
def main():
    # 페이지 제목
    st.title("Loan Approval Prediction Dashboard")

    # 데이터 로드
    df = load_data()

    # 데이터 전처리
    st.subheader("Before Preprocessing")
    st.dataframe(df.head())

    df_preprocessed = preprocess_data(df)

    # 전처리 후 데이터 미리보기
    st.subheader("After Preprocessing")
    st.dataframe(df_preprocessed.head())

    # 데이터 통계량
    st.subheader("Statistical Summary")
    st.write(df_preprocessed.describe())

    # 1. 대출 목적에 따른 평균 대출 금액 시각화 (슬라이더 추가)
    st.subheader("1. 대출 목적에 따른 평균 대출 금액")
    
    # 슬라이더로 대출 금액 범위 선택
    min_loan_amnt = int(df_preprocessed['loan_amnt'].min())
    max_loan_amnt = int(df_preprocessed['loan_amnt'].max())
    loan_amnt_slider = st.slider('대출 금액 범위 선택', min_loan_amnt, max_loan_amnt, (min_loan_amnt, max_loan_amnt))

    # 대출 금액 필터링
    filtered_data = df_preprocessed[(df_preprocessed['loan_amnt'] >= loan_amnt_slider[0]) & (df_preprocessed['loan_amnt'] <= loan_amnt_slider[1])]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='loan_intent', y='loan_amnt', data=filtered_data, ax=ax1, ci=None)
    ax1.set_title('Average Loan Amount by Loan Intent')
    ax1.set_xlabel('Loan Intent')
    ax1.set_ylabel('Average Loan Amount')
    st.pyplot(fig1)

    # 2. 대출 등급에 따른 이자율 분포 시각화 (슬라이더 추가)
    st.subheader("2. 대출 등급에 따른 이자율 분포 시각화")

    # 슬라이더로 이자율 범위 선택
    min_loan_int_rate = float(df_preprocessed['loan_int_rate'].min())
    max_loan_int_rate = float(df_preprocessed['loan_int_rate'].max())
    loan_int_rate_slider = st.slider('이자율 범위 선택', min_loan_int_rate, max_loan_int_rate, (min_loan_int_rate, max_loan_int_rate))

    # 이자율 필터링
    filtered_data2 = df_preprocessed[(df_preprocessed['loan_int_rate'] >= loan_int_rate_slider[0]) & (df_preprocessed['loan_int_rate'] <= loan_int_rate_slider[1])]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='loan_grade', y='loan_int_rate', data=filtered_data2, ax=ax2)
    ax2.set_title('Loan Interest Rate by Loan Grade')
    ax2.set_xlabel('Loan Grade')
    ax2.set_ylabel('Loan Interest Rate (%)')
    st.pyplot(fig2)

    # 3. 대출 승인 여부에 따른 주택 소유 상태 시각화
    st.subheader("3. 대출 승인 여부에 따른 주택 소유 상태 시각화")

    # 대출 승인 여부 필터링 (슬라이더 또는 선택 옵션)
    loan_status_filter = st.selectbox('대출 승인 여부 선택', ['전체', '승인(1)', '미승인(0)'])
    if loan_status_filter == '승인(1)':
        filtered_data3 = df_preprocessed[df_preprocessed['loan_status'] == 1]
    elif loan_status_filter == '미승인(0)':
        filtered_data3 = df_preprocessed[df_preprocessed['loan_status'] == 0]
    else:
        filtered_data3 = df_preprocessed

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='person_home_ownership', hue='loan_status', data=filtered_data3, ax=ax3)
    ax3.set_title('Home Ownership by Loan Status')
    ax3.set_xlabel('Home Ownership')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

# 실행
if __name__ == "__main__":
    main()
