<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet
  version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:wix="http://schemas.microsoft.com/wix/2006/wi"
  exclude-result-prefixes="wix">
  <xsl:output method="xml" indent="yes" />
  <xsl:strip-space elements="*" />

  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()" />
    </xsl:copy>
  </xsl:template>

  <xsl:template match="wix:File/@KeyPath" />

  <xsl:template match="wix:Component[wix:File]">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()" />
      <wix:RegistryValue
        Root="HKCU"
        Key="Software\Lucerna Labs\Atom Harness\Components"
        Name="{@Id}"
        Type="integer"
        Value="1"
        KeyPath="yes" />
    </xsl:copy>
  </xsl:template>

  <xsl:template match="wix:Directory">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()" />
      <wix:Component Guid="*">
        <xsl:attribute name="Id">
          <xsl:value-of select="concat('Cleanup_', @Id)" />
        </xsl:attribute>
        <wix:RemoveFolder On="uninstall">
          <xsl:attribute name="Id">
            <xsl:value-of select="concat('Remove_', @Id)" />
          </xsl:attribute>
        </wix:RemoveFolder>
        <wix:RegistryValue
          Root="HKCU"
          Key="Software\Lucerna Labs\Atom Harness\Folders"
          Name="{@Id}"
          Type="integer"
          Value="1"
          KeyPath="yes" />
      </wix:Component>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="wix:ComponentGroup[@Id='AppFiles']">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()" />
      <xsl:for-each
        select="/wix:Wix/wix:Fragment/wix:DirectoryRef//wix:Directory">
        <wix:ComponentRef>
          <xsl:attribute name="Id">
            <xsl:value-of select="concat('Cleanup_', @Id)" />
          </xsl:attribute>
        </wix:ComponentRef>
      </xsl:for-each>
    </xsl:copy>
  </xsl:template>
</xsl:stylesheet>
