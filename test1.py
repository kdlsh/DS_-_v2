import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import kendalltau
from scipy.stats import spearmanr


# ‘더치페이 요청에 대한 응답률이 높을수록 더치페이 서비스를 더 많이 사용한다.’
# 라는 가설을 통계적으로 검정해주세요.

# "더치페이 요청에 대한 응답률 정도" 은 "CHECK status는 제외하고 SEND 의 rate 순위" 로 측정
# "더치페이 서비스 이용 정도" 는 "정해진 시간단위의 더치페이 요청 횟수 순위"로 측정
# 비모수 상관 검정 -> Spearman correlation, Kendall correlation


def load_data():
    """
    # Read CSV files
    """
    claim_df = pd.read_csv("dutchpay_claim.csv")
    claim_detail_df = pd.read_csv("dutchpay_claim_detail.csv")
    users_df = pd.read_csv("users.csv")
    a_payment_trx_df = pd.read_csv("a_payment_trx.csv")

    return claim_df, claim_detail_df, users_df, a_payment_trx_df


def review_detail_df(claim_detail_df):
    """
    # Review claim_detail status
    # SEND -> 요청받은 돈을 송금함  
    # CHECK -> 요청한 사람  
    # CLAIM -> 요청받은 돈을 주지 않음 
    """
    # Review claim_detail status
    for i, df in enumerate(claim_detail_df.groupby("claim_id")):
        print(df[1])
        if i > 10:
            break

    # CHECK status review
    print(
        claim_detail_df[claim_detail_df["status"] == "CHECK"]["send_amount"]
        .isna()
        .sum()
    )
    # status가 CHECK면 send_amount NaN은 없다.
    print(
        (
            claim_detail_df[claim_detail_df["status"] == "CHECK"]["send_amount"] == 0
        ).sum()
    )
    # status가 CHECK일때 send_amount 0 -> 4,646 케이스
    print(claim_detail_df[claim_detail_df["send_amount"] == 0].shape[0])
    # send_amount 0 인 케이스는 4,646건으로 모두 CHECK


def get_claim_response_df(claim_detail_df):
    """
    # 더치페이 요청에 대한 응답률
    """
    d_df = claim_detail_df.copy()

    # 더치페이 요청자 제외
    claim_response_df = d_df[d_df["status"] != "CHECK"]

    # 결제요청수신자, status 로 groupby count
    claim_response_df = (
        claim_response_df.groupby(["recv_user_id", "status"])
        .count()["claim_detail_id"]
        .to_frame()
    )

    # level 1 transpose to column
    claim_response_df = claim_response_df.stack().unstack(level=1)

    # drop redundant index
    claim_response_df.index = claim_response_df.index.droplevel(1)

    # remove columns name
    claim_response_df.columns.name = ""
    claim_response_df.index.name = "user_id"

    # fill NaN to 0
    claim_response_df = claim_response_df.fillna(0)

    # total count
    claim_response_df["TOTAL"] = claim_response_df["CLAIM"] + claim_response_df["SEND"]

    # 응답률
    claim_response_df["SEND_RATE"] = (
        claim_response_df["SEND"] / claim_response_df["TOTAL"]
    ).round(3)

    # type casting
    dtype_dic = {"CLAIM": "int64", "SEND": "int64", "TOTAL": "int64"}
    claim_response_df = claim_response_df.astype(dtype_dic)

    # drop claim column
    claim_response_df = claim_response_df.drop(columns=["CLAIM", "SEND", "TOTAL"])

    return claim_response_df


def get_total_claim_df(claim_df, users_df, claim_detail_df):
    """
    더치페이 사용정보
    """

    c_df = claim_df.copy()
    u_df = users_df.copy()
    d_df = claim_detail_df.copy()

    # dutchpay user claim df
    c_df = (
        c_df.groupby(["claim_user_id"]).count()["claim_id"].rename("CLAIM").to_frame()
    )
    c_df.index.name = "user_id"

    # total user claim df
    users_user_series = u_df["user_id"]
    clain_detail_user_series = d_df["recv_user_id"]
    total_user_series = users_user_series.append(clain_detail_user_series).rename(
        "user_id"
    )
    total_user_df = total_user_series.to_frame().drop_duplicates()
    total_user_df["CLAIM"] = 0
    total_user_df = total_user_df.set_index("user_id")

    # merge counts
    totla_claim_df = total_user_df + c_df
    totla_claim_df = totla_claim_df.fillna(0)

    return totla_claim_df


def hypo_test(totla_claim_df, claim_response_df):
    """
    #통계 검정
    "더치페이 요청에 대한 응답률 정도" 은 "CHECK status는 제외하고 SEND 의 rate 순위" 로 측정 -> 비모수
    "더치페이 서비스 이용 정도" 는 "정해진 시간단위의 더치페이 요청 횟수 순위"로 측정 -> 비모수

    비모수 상관 검정 -> Spearman correlation, Kendall correlation

    귀무가설 H0 : 응답률 정도와 서비스 이용 정도는 상관관계가 없다
    대립가설 H1 : 응답률 정도와 서비스 이용 정도는 상관관계가 있다
    """

    # inner join
    test_df = pd.merge(
        totla_claim_df, claim_response_df, left_index=True, right_index=True
    )

    # kendall tau
    print(kendalltau(test_df["CLAIM"], test_df["SEND_RATE"]))

    # spearman
    print(spearmanr(test_df["CLAIM"], test_df["SEND_RATE"]))


if __name__ == "__main__":

    # Read CSV files
    claim_df, claim_detail_df, users_df, a_payment_trx_df = load_data()

    # Review claim_detail status
    # review_detail_df(claim_detail_df)

    # 더치페이 요청에 대한 응답률
    claim_response_df = get_claim_response_df(claim_detail_df)

    # 더치페이 사용정보
    totla_claim_df = get_total_claim_df(claim_df, users_df, claim_detail_df)

    # 통계 검정
    hypo_test(totla_claim_df, claim_response_df)
