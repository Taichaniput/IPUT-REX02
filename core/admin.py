from django.contrib import admin

# financial/admin.py

from django.contrib import admin
from .models import EdinetDocuments, FinancialData, FinancialDataValidated


@admin.register(EdinetDocuments)
class EdinetDocumentAdmin(admin.ModelAdmin):
    list_display = ['doc_id', 'filer_name', 'period_end', 'is_xbrl_parsed']
    list_filter = ['is_xbrl_parsed', 'doc_type_code']
    search_fields = ['doc_id', 'edinet_code', 'filer_name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(FinancialData)
class FinancialDataAdmin(admin.ModelAdmin):
    list_display = ['document_id', 'filer_name', 'edinet_code','fiscal_year', 'net_sales_display']
    list_filter = ['fiscal_year']
    search_fields = ['edinet_code', 'filer_name']
    
    def net_sales_display(self, obj):
        if obj.net_sales:
            return f"{obj.net_sales / 100000000:.1f}億円"
        return "-"
    net_sales_display.short_description = "売上高"
    
    # 読み取り専用にする
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False