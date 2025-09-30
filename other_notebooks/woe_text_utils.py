import matplotlib.pyplot as plt
import pandas as pd


def explain_prediction_for_class(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    classifiers: dict,
    woe_mappings: dict,
    label_to_intent: dict,
    instance_index: int,
):
    """
    Explains the prediction for a specific instance by identifying the class with the highest probability
    and generating a WOE-based explanation for that class.

    Parameters:
    - df (pd.DataFrame): The original DataFrame containing the text data.
    - text_col (str): The name of the column containing the text.
    - label_col (str): The name of the column containing the labels.
    - classifiers (dict): Dictionary of trained models for each class.
    - woe_mappings (dict): A dictionary of WOE mappings for each class.
    - label_to_intent (dict): A dictionary mapping label numbers to intent names.
    - instance_index (int): The index of the instance to explain.

    Returns:
    - explanation_df (pd.DataFrame): A DataFrame containing tokens and their WOE contributions.
    - class_with_highest_proba (int): The class with the highest prediction probability.
    """
    # Get the instance's text
    instance_text = df.loc[instance_index, text_col]

    # Tokenize the text
    tokenized_text = instance_text.split()

    # Initialize variables to store the highest probability and corresponding class
    highest_proba = 0
    class_with_highest_proba = None

    # Loop through each class and classifier to find the class with the highest prediction probability
    for cls, model in classifiers.items():
        # Retrieve WOE mapping for the class
        woe_mapping = woe_mappings[cls]

        # Filter to the tokens present in the instance
        present_tokens = [token for token in tokenized_text if token in woe_mapping]

        # Get WOE scores for the present tokens
        token_woe_scores = {token: woe_mapping[token] for token in present_tokens}

        # Prepare the data for prediction
        instance_features = (
            pd.DataFrame([token_woe_scores]).reindex(columns=woe_mapping.keys()).fillna(0)
        )

        # Predict the probability for the class using the specific model
        predicted_proba = model.predict_proba(instance_features)[0][1]
        
        # Check if this class has the highest probability or is the first with proba > 0.5
        if predicted_proba > highest_proba:
            highest_proba = predicted_proba
            class_with_highest_proba = cls

    # Now that we have the class with the highest probability, generate the explanation for it
    woe_mapping = woe_mappings[class_with_highest_proba]
    present_tokens = [token for token in tokenized_text if token in woe_mapping]
    token_woe_scores = {token: woe_mapping[token] for token in present_tokens}
    sorted_token_woe = sorted(token_woe_scores.items(), key=lambda item: item[1], reverse=True)
    explanation_df = pd.DataFrame(sorted_token_woe, columns=["Token", "WOE Score"])

    # Retrieve the true label for the instance
    true_label = df.loc[instance_index, label_col]
    predicted_intent = label_to_intent[class_with_highest_proba]

    # Print the results
    print(f"True label for the instance: {true_label} ({label_to_intent[true_label]})")
    print(
        f"Class with highest prediction probability: {class_with_highest_proba} ({predicted_intent})"
    )
    print(f"Highest prediction probability: {highest_proba:.4f}")
    print(f"Top contributing tokens for class {class_with_highest_proba}:")

    return explanation_df, class_with_highest_proba


def plot_woe_contributions(explanation_df, class_to_explain, true_label, label_to_intent):
    """
    Plots the WOE contributions for the top tokens in the explanation DataFrame.

    Parameters:
    - explanation_df (pd.DataFrame): DataFrame containing tokens and WOE scores.
    - class_to_explain (int): The class (label) for which the explanation is generated.
    - true_label (int): The true label of the instance.
    - label_to_intent (dict): A dictionary mapping label numbers to intent names.

    Returns:
    - None: Displays the plot.
    """
    # Get the intent name using the class label
    intent_name = label_to_intent.get(class_to_explain, f"Class {class_to_explain}")

    # Set font to 0xProto if available
    plt.rcParams["font.family"] = "0xProto"

    # Determine bar colors based on WOE score: red for negative, blue for positive
    colors = ["#e64980" if woe < 0 else "#008eff" for woe in explanation_df["WOE Score"]]

    # Create the plot
    plt.figure(figsize=(9, 5), dpi=150)
    plt.bar(explanation_df["Token"], explanation_df["WOE Score"], color=colors)
    plt.xlabel("Token")
    plt.ylabel("WOE Score")
    plt.title(f"Predicted: {intent_name}\nTrue: {label_to_intent[true_label]}")

    # Disable top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Add horizontal line at 0
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.xticks(rotation=90)

    plt.show()


label_to_intent = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "Refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment?",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal",
}
