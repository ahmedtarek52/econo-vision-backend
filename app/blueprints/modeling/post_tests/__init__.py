# app/blueprints/modeling/post_tests/__init__.py

# 1. Time Series Tests (VAR, VECM, ARIMA, GARCH)
# تم نقل الكود المطور والمنسق جمالياً إلى هذا الملف
from .time_series import (
    run_var_stability,
    run_irf,
    run_fevd,
    run_var_normality,
    run_var_autocorr,
    run_ljung_box,
    run_arch_lm,
    run_ljung_box_std
)

# 2. Econometric Tests (OLS, Panel, ARDL)
# يفترض وجود ملف 'econometric.py' يحتوي على هذه الدوال
from .econometric import (
    run_ramsey_reset,
    run_white_test,
    run_jarque_bera_resid,
    run_durbin_watson,
    run_cusum_plot,
    run_chow_test,
    run_panel_serial_corr,
    run_panel_hetero,
    run_classification_report,
    run_hosmer_lemeshow
)

# 3. Machine Learning & Causal Tests (DML, Random Forest)
# يفترض وجود ملف 'ml_causal.py' يحتوي على هذه الدوال
from .ml_causal import (
    run_ml_residuals_plot,
    run_dml_sensitivity,
    run_dml_heterogeneity
)

# 4. General Utilities (Forecasting, Equations)
# يفترض وجود ملف 'general.py' يحتوي على هذه الدوال
from .general import (
    run_forecast,
    run_system_equations
)