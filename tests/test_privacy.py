from finaudit_rag.privacy import mask_sensitive_text


def test_masks_supported_identifiers_without_removing_amounts_or_circular_ids() -> None:
    original = (
        "PAN ABCPD1234E, Aadhaar 2345 6789 0123, account number: 0123456789, "
        "analyst@example.org, revenue INR 1,36,592 crore, SEBI/HO/MIRSD/CIR/2025/80"
    )
    masked = mask_sensitive_text(original)
    assert masked.text == (
        "PAN [PAN], Aadhaar [AADHAAR], account number: [ACCOUNT], "
        "[EMAIL], revenue INR 1,36,592 crore, SEBI/HO/MIRSD/CIR/2025/80"
    )
    assert masked.replacements == 4


def test_masks_sentence_final_addresses_and_table_account_numbers() -> None:
    masked = mask_sensitive_text("Contact analyst@example.org.\nAccount number | 1234567890")
    assert masked.text == "Contact [EMAIL].\nAccount number | [ACCOUNT]"
    assert masked.replacements == 2
