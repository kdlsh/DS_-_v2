import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# 더치페이를 요청한 유저 중 a 가맹점에서 2019년 12월에 1만원 이상 결제한 유저를
# 대상으로 리워드를 지급하려고 합니다. 리워드 지급 대상자 user_id를 추출하는 SQL
# 쿼리를 작성해주세요.
# - 2019년 12월 결제분 중 취소를 반영한 순결제금액 1만원 이상인 유저만을 대상으로 함
# - 취소 반영기간은 2020년 2월까지로 함

# csv file 을 sql-db화 시킨후 query 수행
# 1. 2020년2월28일 까지 데이터 사용
# 2. 더치페이 요청한 유저로 제한
# 3. 결제취소 포함 transaction_id 제외
# 4. 2019년12월01일 ~ 2019년12월31일 포함
# 5. 총 결제금액 >= 10000원 포함


def load_db():
    # sqlite engine
    engine = create_engine("sqlite://", echo=False)

    # read csv files to sqlite db
    pd.read_csv("dutchpay_claim.csv").to_sql("dutchpay_claim", con=engine)
    pd.read_csv("dutchpay_claim_detail.csv").to_sql("dutchpay_claim_detail", con=engine)
    pd.read_csv("users.csv").to_sql("users", con=engine)
    pd.read_csv("a_payment_trx.csv").to_sql("a_payment_trx", con=engine)

    return engine


if __name__ == "__main__":
    engine = load_db()

    # GROUP BY transaction_id \
    sql = """SELECT 
                DISTINCT(user_id)
            FROM 
                a_payment_trx
            WHERE 
                transacted_at <= '2020-02-28' and
                user_id in (SELECT DISTINCT(claim_user_id) FROM dutchpay_claim)
            GROUP BY 
                transaction_id
            HAVING 
                GROUP_CONCAT(payment_action_type, ',') NOT LIKE "%CANCEL%" and        
                transacted_at BETWEEN '2019-12-01' and '2019-12-31' and
                SUM(amount)>= 10000
        """

    query_list = engine.execute(sql).fetchall()

    print(len(query_list))
    # 686명의 user_id
