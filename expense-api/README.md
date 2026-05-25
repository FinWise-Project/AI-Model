Cara test user_id baru dengan data ≥ 3 bulan → auto-train lalu prediksi.

{
  "user_id": "USER_021", (Dari USER 0_21 - seterusnya)
  "transactions": [
    
    {"date": "2024-01-01", "amount": 6000000, "type": "income", "category": "Salary", "subcategory": "gaji bulanan", "payment_method": "debit", "description": "Pembayaran gaji bulanan"},
    {"date": "2024-01-05", "amount": 45000, "type": "expense", "category": "Food", "subcategory": "warung", "payment_method": "cash", "description": "Pembayaran warung"},
    {"date": "2024-01-08", "amount": 35000, "type": "expense", "category": "Food", "subcategory": "kopi", "payment_method": "cash", "description": "Pembayaran kopi"},
    {"date": "2024-01-10", "amount": 150000, "type": "expense", "category": "Bills", "subcategory": "listrik", "payment_method": "debit", "description": "Pembayaran listrik"},
    {"date": "2024-01-12", "amount": 100000, "type": "expense", "category": "Bills", "subcategory": "pulsa", "payment_method": "e-wallet", "description": "Pembayaran pulsa"},
    {"date": "2024-01-15", "amount": 50000, "type": "expense", "category": "Transport", "subcategory": "ojek online", "payment_method": "e-wallet", "description": "Pembayaran ojek online"},
    {"date": "2024-01-18", "amount": 120000, "type": "expense", "category": "Shopping", "subcategory": "minimarket", "payment_method": "cash", "description": "Pembayaran minimarket"},
    {"date": "2024-01-22", "amount": 50000, "type": "expense", "category": "Entertainment", "subcategory": "netflix", "payment_method": "e-wallet", "description": "Pembayaran netflix"},
    {"date": "2024-01-25", "amount": 67000, "type": "expense", "category": "Food", "subcategory": "restoran", "payment_method": "e-wallet", "description": "Pembayaran restoran"},

    {"date": "2024-02-01", "amount": 6000000, "type": "income", "category": "Salary", "subcategory": "gaji bulanan", "payment_method": "debit", "description": "Pembayaran gaji bulanan"},
    {"date": "2024-02-04", "amount": 40000, "type": "expense", "category": "Food", "subcategory": "warung", "payment_method": "cash", "description": "Pembayaran warung"},
    {"date": "2024-02-07", "amount": 30000, "type": "expense", "category": "Transport", "subcategory": "bensin", "payment_method": "cash", "description": "Pembayaran bensin"},
    {"date": "2024-02-10", "amount": 150000, "type": "expense", "category": "Bills", "subcategory": "internet", "payment_method": "debit", "description": "Pembayaran internet"},
    {"date": "2024-02-13", "amount": 80000, "type": "expense", "category": "Food", "subcategory": "fast food", "payment_method": "e-wallet", "description": "Pembayaran fast food"},
    {"date": "2024-02-16", "amount": 100000, "type": "expense", "category": "Shopping", "subcategory": "pakaian", "payment_method": "debit", "description": "Pembayaran pakaian"},
    {"date": "2024-02-20", "amount": 15000, "type": "expense", "category": "Transport", "subcategory": "parkir", "payment_method": "cash", "description": "Pembayaran parkir"},
    {"date": "2024-02-23", "amount": 50000, "type": "expense", "category": "Entertainment", "subcategory": "spotify", "payment_method": "e-wallet", "description": "Pembayaran spotify"},
    {"date": "2024-02-26", "amount": 35000, "type": "expense", "category": "Food", "subcategory": "kopi", "payment_method": "cash", "description": "Pembayaran kopi"},

    {"date": "2024-03-01", "amount": 6000000, "type": "income", "category": "Salary", "subcategory": "gaji bulanan", "payment_method": "debit", "description": "Pembayaran gaji bulanan"},
    {"date": "2024-03-03", "amount": 45000, "type": "expense", "category": "Food", "subcategory": "warung", "payment_method": "cash", "description": "Pembayaran warung"},
    {"date": "2024-03-06", "amount": 150000, "type": "expense", "category": "Bills", "subcategory": "listrik", "payment_method": "debit", "description": "Pembayaran listrik"},
    {"date": "2024-03-09", "amount": 50000, "type": "expense", "category": "Transport", "subcategory": "ojek online", "payment_method": "e-wallet", "description": "Pembayaran ojek online"},
    {"date": "2024-03-12", "amount": 67000, "type": "expense", "category": "Food", "subcategory": "restoran", "payment_method": "e-wallet", "description": "Pembayaran restoran"},
    {"date": "2024-03-15", "amount": 120000, "type": "expense", "category": "Shopping", "subcategory": "skincare", "payment_method": "e-wallet", "description": "Pembayaran skincare"},
    {"date": "2024-03-20", "amount": 30000, "type": "expense", "category": "Others", "subcategory": "donasi", "payment_method": "cash", "description": "Pembayaran donasi"},
    {"date": "2024-03-25", "amount": 50000, "type": "expense", "category": "Entertainment", "subcategory": "bioskop", "payment_method": "cash", "description": "Pembayaran bioskop"}
  ],
  "onboarding_estimate": null
}