data_dict = {
    "ExternalRiskEstimate": {
        "standardized_attribute_name": "external_risk_estimate",
        "causal_knowledge": -1,
    },
    "MSinceOldestTradeOpen": {
        "standardized_attribute_name": "months_since_oldest_trade_open",
        "causal_knowledge": -1,
    },
    "MSinceMostRecentTradeOpen": {
        "standardized_attribute_name": "months_since_most_recent_trade_open",
        "causal_knowledge": -1,
    },
    "AverageMInFile": {
        "standardized_attribute_name": "average_months_in_file",
        "causal_knowledge": -1,
    },
    "NumSatisfactoryTrades": {
        "standardized_attribute_name": "num_satisfactory_trades",
        "causal_knowledge": -1,
    },
    "NumTrades60Ever2DerogPubRec": {
        "standardized_attribute_name": "num_trades_60_ever_2_derog_pub_rec",
        "causal_knowledge": 1,
    },
    "NumTrades90Ever2DerogPubRec": {
        "standardized_attribute_name": "num_trades_90_ever_2_derog_pub_rec",
        "causal_knowledge": 1,
    },
    "PercentTradesNeverDelq": {
        "standardized_attribute_name": "percent_trades_never_delq",
        "causal_knowledge": -1,
    },
    "MSinceMostRecentDelq": {
        "standardized_attribute_name": "months_since_most_recent_delq",
        "causal_knowledge": -1,
    },
    "NumTradesOpeninLast12M": {
        "standardized_attribute_name": "num_trades_open_in_last_12m",
        "causal_knowledge": 1,
    },
    "MSinceMostRecentInqexcl7days": {
        "standardized_attribute_name": "months_since_most_recent_inqexcl7days",
        "causal_knowledge": -1,
    },
    "NumInqLast6M": {
        "standardized_attribute_name": "num_inq_last_6m",
        "causal_knowledge": 1,
    },
    "NumInqLast6Mexcl7days": {
        "standardized_attribute_name": "num_inq_last_6m_excl7days",
        "causal_knowledge": 1,
    },
    "NetFractionRevolvingBurden": {
        "standardized_attribute_name": "net_fraction_revolving_burden",
        "causal_knowledge": 1,
    },
    "NetFractionInstallBurden": {
        "standardized_attribute_name": "net_fraction_install_burden",
        "causal_knowledge": 1,
    },
    "NumBank2NatlTradesWHighUtilization": {
        "standardized_attribute_name": "num_bank_2_natl_trades_w_high_utilization",
        "causal_knowledge": 1,
    },
    "emp_length": {"standardized_attribute_name": "emp_length", "causal_knowledge": -1},
    "annual_income": {
        "standardized_attribute_name": "annual_income",
        "causal_knowledge": 1,
    },
    "debt_to_income": {
        "standardized_attribute_name": "debt_to_income",
        "causal_knowledge": 1,
    },
    "total_credit_limit": {
        "standardized_attribute_name": "total_credit_limit",
        "causal_knowledge": 1,
    },
    "total_credit_utilized": {
        "standardized_attribute_name": "total_credit_utilized",
        "causal_knowledge": 1,
    },
    "num_historical_failed_to_pay": {
        "standardized_attribute_name": "num_historical_failed_to_pay",
        "causal_knowledge": 1,
    },
    "current_installment_accounts": {
        "standardized_attribute_name": "current_installment_accounts",
        "causal_knowledge": 1,
    },
    "num_total_cc_accounts": {
        "standardized_attribute_name": "num_total_cc_accounts",
        "causal_knowledge": 1,
    },
    "num_open_cc_accounts": {
        "standardized_attribute_name": "num_open_cc_accounts",
        "causal_knowledge": 1,
    },
    "num_cc_carrying_balance": {
        "standardized_attribute_name": "num_cc_carrying_balance",
        "causal_knowledge": 1,
    },
    "num_mort_accounts": {
        "standardized_attribute_name": "num_mort_accounts",
        "causal_knowledge": -1,
    },
    "account_never_delinq_percent": {
        "standardized_attribute_name": "account_never_delinq_percent",
        "causal_knowledge": 1,
    },
    "loan_amount": {
        "standardized_attribute_name": "loan_amount",
        "causal_knowledge": 1,
    },
    "balance": {"standardized_attribute_name": "balance", "causal_knowledge": 1},
    "paid_total": {"standardized_attribute_name": "paid_total", "causal_knowledge": -1},
    "age": {"standardized_attribute_name": "age", "causal_knowledge": -1},
    "DebtRatio": {"standardized_attribute_name": "debt_ratio", "causal_knowledge": -1},
    "MonthlyIncome": {
        "standardized_attribute_name": "monthly_income",
        "causal_knowledge": -1,
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "standardized_attribute_name": "number_of_open_credit_lines_and_loans",
        "causal_knowledge": 1,
    },
    "NumberRealEstateLoansOrLines": {
        "standardized_attribute_name": "number_real_estate_loans_or_lines",
        "causal_knowledge": 1,
    },
    "NumberOfDependents": {
        "standardized_attribute_name": "number_of_dependents",
        "causal_knowledge": 1,
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "standardized_attribute_name": "revolving_utilization_of_unsecured_lines",
        "causal_knowledge": 1,
    },
}

special_codes = [-9, -8, -7]