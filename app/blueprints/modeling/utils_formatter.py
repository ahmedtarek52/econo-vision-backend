import numpy as np
import pandas as pd

def format_to_stargazer_html(model_results_dict, dep_var_name="Dependent Variable"):
    """
    Formats single or multiple econometric model results into a publication-ready
    Stargazer-style HTML table.
    Expects a dict: {'Pooled OLS': model_1, 'Fixed Effects (FE)': model_2, ...}
    """
    try:
        # 1. تجميع كل المتغيرات المستقلة الفريدة من كل النماذج الممررة
        all_indep_vars = []
        for name, res in model_results_dict.items():
            if res is not None:
                # دعم كلاً من statsmodels و linearmodels
                vars_list = res.params.index.tolist()
                for v in vars_list:
                    if v not in all_indep_vars and v != 'const':
                        all_indep_vars.append(v)
                        
        # وضع الثابت (const) في نهاية قائمة المتغيرات دائماً كالعرف الأكاديمي
        if any('const' in res.params.index for res in model_results_dict.values() if res is not None):
            all_indep_vars.append('const')

        # 2. بناء هيكل جدول HTML بستايل النشر الدولي
        html = '<table class="stargazer-academic-table">'
        html += f'<caption>Model Estimation Matrix (Dependent Variable: {dep_var_name})</caption>'
        
        # خط الرأس العلوي
        html += '<thead>'
        html += '<tr class="top-border"><th>Regressor</th>'
        for model_name in model_results_dict.keys():
            html += f'<th>{model_name}</th>'
        html += '</tr>'
        
        html += '<tr class="bottom-border"><td></td>'
        for res in model_results_dict.values():
            estimator_name = res.__class__.__name__ if res is not None else "N/A"
            if "Pooled" in estimator_name: estimator_name = "Pooled OLS"
            elif "Panel" in estimator_name: estimator_name = "Fixed Effects"
            elif "Random" in estimator_name: estimator_name = "Random Effects"
            html += f'<td><span class="estimator-sub font-mono text-xs text-gray-500">({estimator_name})</span></td>'
        html += '</tr></thead><tbody>'

        # 3. حشو البيانات: صف للمعامل وصف متداخل تحت للـ T-Stat
        for var in all_indep_vars:
            # الاسم النظيف للمتغير
            display_var = "Constant" if var == 'const' else var
            
            # الصف الأول: المعامل والنجوم
            html += f'<tr><td class="var-name font-semibold text-left">{display_var}</td>'
            for res in model_results_dict.values():
                if res is not None and var in res.params.index:
                    coef = res.params[var]
                    pval = res.pvalues[var]
                    
                    # حساب النجوم
                    stars = ""
                    if pval <= 0.01: stars = "***"
                    elif pval <= 0.05: stars = "**"
                    elif pval <= 0.1: stars = "*"
                    
                    html += f'<td>{coef:.4f}{stars}</td>'
                else:
                    html += '<td></td>'
            html += '</tr>'

            # الصف الثاني: الـ T-Statistic المقوس تحت المعامل مباشرة
            html += '<tr class="t-stat-row"><td></td>'
            for res in model_results_dict.values():
                tstats = getattr(res, 'tvalues', getattr(res, 'tstats', None)) if res is not None else None
                if res is not None and tstats is not None and var in tstats.index:
                    t_stat = tstats[var]
                    html += f'<td><span class="text-gray-500">({t_stat:.4f})</span></td>'
                else:
                    html += '<td></td>'
            html += '</tr>'

        # 4. إضافة الإحصائيات الكلية للنموذج في الأسفل (Model Diagnostics summary)
        html += '<tr class="top-border"><td class="font-medium text-left">Observations</td>'
        for res in model_results_dict.values():
            nobs = getattr(res, 'nobs', 'N/A')
            html += f'<td>{nobs}</td>'
        html += '</tr>'

        html += '<tr><td class="font-medium text-left">R-squared</td>'
        for res in model_results_dict.values():
            rsq = getattr(res, 'rsquared', None)
            rsq_str = f"{rsq:.4f}" if rsq is not None else "N/A"
            html += f'<td>{rsq_str}</td>'
        html += '</tr>'
        
        # تفاصيل الـ F-Statistic
        html += '<tr class="bottom-border"><td class="font-medium text-left">F-Statistic / Prob</td>'
        for res in model_results_dict.values():
            f_stat = getattr(res, 'f_statistic', None)
            if f_stat is not None and hasattr(f_stat, 'stat'):
                html += f'<td>{f_stat.stat:.2f} <span class="text-xs">({f_stat.pvalue:.4f})</span></td>'
            elif hasattr(res, 'fvalue'):
                html += f'<td>{res.fvalue:.2f} <span class="text-xs">({res.f_pvalue:.4f})</span></td>'
            else:
                html += '<td>N/A</td>'
        html += '</tr>'

        html += '</tbody></table>'
        # ملاحظة النجوم القياسية في تذييل الجدول
        html += '<div class="stargazer-legend text-xs mt-2 text-gray-500 italic text-left">Note: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. T-statistics reported in parentheses.</div>'
        
        return html
    except Exception as e:
        import traceback
        print("Stargazer error:")
        traceback.print_exc()
        return f"<p class='text-red-500'>Failed to format Stargazer Table: {str(e)}</p>"