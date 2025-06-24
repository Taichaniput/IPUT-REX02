# financial/templatetags/__init__.py
# 空ファイル

# financial/templatetags/financial_filters.py

from django import template

register = template.Library()


@register.filter
def divisibleby(value, arg):
    """数値を指定された値で割る"""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError, TypeError):
        return value


@register.filter
def mul(value, arg):
    """数値に指定された値を掛ける"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value


@register.filter
def to_billion(value):
    """数値を億円単位に変換"""
    try:
        return float(value) / 100000000
    except (ValueError, TypeError):
        return value


@register.filter
def percentage(value):
    """小数をパーセンテージ表示に変換"""
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "-"


@register.filter
def sub(value, arg):
    """引き算"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return value


@register.filter  
def div(value, arg):
    """割り算"""
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def get_feature_label(feature):
    """特徴量の日本語ラベル"""
    labels = {
        'net_assets': '純資産',
        'total_assets': '総資産',
        'net_income': '純利益',
        'r_and_d_expenses': '研究開発費',
        'number_of_employees': '従業員数'
    }
    return labels.get(feature, feature)