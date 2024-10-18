# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# 데이터 로드 함수 (캐시 사용)
@st.cache_data
def load_data():
    # 데이터 로드
    df_train = pd.read_csv('data/train.csv')
    return df_train

# 메인 대시보드 함수
def main():
    # 페이지 제목
    st.title("Loan Approval Prediction Dashboard")

    # 데이터 로드
    data = load_data()

    # 데이터프레임 미리보기
    st.subheader("Data Overview")
    st.dataframe(data.head())

    # 데이터 통계량
    st.subheader("Statistical Summary")
    st.write(data.describe())

    # 1. 대출 목적에 따른 평균 대출 금액 시각화
    st.subheader("1. 대출 목적에 따른 평균 대출 금액")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='loan_intent', y='loan_amnt', data=data, ax=ax1, ci=None)
    ax1.set_title('Average Loan Amount by Loan Intent')
    ax1.set_xlabel('Loan Intent')
    ax1.set_ylabel('Average Loan Amount')
    st.pyplot(fig1)

    # 2. 대출 등급에 따른 이자율 분포 시각화
    st.subheader("2. 대출 등급에 따른 이자율 분포 시각화")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='loan_grade', y='loan_int_rate', data=data, ax=ax2)
    ax2.set_title('Loan Interest Rate by Loan Grade')
    ax2.set_xlabel('Loan Grade')
    ax2.set_ylabel('Loan Interest Rate (%)')
    st.pyplot(fig2)

    # 3. 대출 승인 여부에 따른 주택 소유 상태 시각화
    st.subheader("3. 대출 승인 여부에 따른 주택 소유 상태 시각화")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='person_home_ownership', hue='loan_status', data=data, ax=ax3)
    ax3.set_title('Home Ownership by Loan Status')
    ax3.set_xlabel('Home Ownership')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

# 실행
if __name__ == "__main__":
    main()
