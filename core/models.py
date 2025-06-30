# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.contrib.auth.models import User


class EdinetDocuments(models.Model):
    doc_id = models.CharField(primary_key=True, max_length=10)
    edinet_code = models.CharField(max_length=10, blank=True, null=True)
    filer_name = models.CharField(max_length=255, blank=True, null=True)
    doc_type_code = models.CharField(max_length=10, blank=True, null=True)
    period_end = models.DateField(blank=True, null=True)
    submit_datetime = models.DateTimeField(blank=True, null=True)
    is_xbrl_parsed = models.BooleanField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'edinet_documents'

    def __str__(self):
        return f"{self.filter_name} {(self.doc_id)}"


class FinancialData(models.Model):
    document = models.OneToOneField(EdinetDocuments, models.DO_NOTHING, primary_key=True)
    edinet_code = models.CharField(max_length=10, blank=True, null=True)
    filer_name = models.CharField(max_length=255, blank=True, null=True)
    fiscal_year = models.IntegerField(blank=True, null=True)
    net_assets = models.BigIntegerField(blank=True, null=True)
    total_assets = models.BigIntegerField(blank=True, null=True)
    net_sales = models.BigIntegerField(blank=True, null=True)
    operating_income = models.BigIntegerField(blank=True, null=True)
    ordinary_income = models.BigIntegerField(blank=True, null=True)
    net_income = models.BigIntegerField(blank=True, null=True)
    operating_cash_flow = models.BigIntegerField(blank=True, null=True)
    r_and_d_expenses = models.BigIntegerField(blank=True, null=True)
    number_of_employees = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'financial_data'
        ordering = ['-fiscal_year']

    def __str__(self):
        return f"{self.filter_name} {self.fical_year}年度"
    

    @property
    def net_sales_billion(self):
        """売上高（億円）"""
        return self.net_sales / 100000000 if self.net_sales else None
    
    @property
    def roe(self):
        """ROE（自己資本利益率）"""
        if self.net_assets and self.net_assets > 0 and self.net_income:
            return self.net_income / self.net_assets
        return None

class FinancialDataValidated(models.Model):
    document = models.OneToOneField(EdinetDocuments, models.DO_NOTHING, primary_key=True)
    edinet_code = models.CharField(max_length=10)
    filer_name = models.CharField(max_length=255)
    fiscal_year = models.IntegerField(blank=True, null=True)
    net_assets = models.BigIntegerField()
    total_assets = models.BigIntegerField()
    net_sales = models.BigIntegerField()
    operating_income = models.BigIntegerField()
    ordinary_income = models.BigIntegerField()
    net_income = models.BigIntegerField()
    total_liabilities = models.BigIntegerField()
    operating_cash_flow = models.BigIntegerField()
    r_and_d_expenses = models.BigIntegerField()
    number_of_employees = models.IntegerField()
    confidence_score = models.IntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'financial_data_validated'
        ordering = ['-fiscal_year']


class UserProfile(models.Model):
    """ユーザープロフィール"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=20, blank=True, null=True, verbose_name="学籍番号")
    university = models.CharField(max_length=100, blank=True, null=True, verbose_name="大学名")
    department = models.CharField(max_length=100, blank=True, null=True, verbose_name="学部・学科")
    graduation_year = models.IntegerField(blank=True, null=True, verbose_name="卒業予定年")
    interests = models.TextField(blank=True, null=True, verbose_name="興味のある業界・職種")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "ユーザープロフィール"
        verbose_name_plural = "ユーザープロフィール"

    def __str__(self):
        return f"{self.user.username}のプロフィール"
