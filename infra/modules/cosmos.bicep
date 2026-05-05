// Cosmos DB NoSQL serverless account, database, and containers.
// Supports bring-your-own (existing) account via useExisting flag.

@description('Whether to use an existing Cosmos account.')
param useExisting bool = false

@description('Name of the new Cosmos account (when useExisting=false).')
param accountName string

@description('Name of an existing Cosmos account (when useExisting=true).')
param existingAccountName string = ''

@description('Resource group of the existing Cosmos account (when useExisting=true). If empty, current RG is used.')
param existingResourceGroup string = ''

@description('Azure region for new account.')
param location string

@description('Database name (created if missing).')
param databaseName string = 'ai_memory'

@description('Whether to also create the Durable Function support containers (leases, counter).')
param deployFunctionContainers bool = true

@description('Tags to apply to created resources.')
param tags object = {}

// --- Account ---------------------------------------------------------------

resource newAccount 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = if (!useExisting) {
  name: accountName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    disableLocalAuth: false
    publicNetworkAccess: 'Enabled'
  }
}

resource existingAccount 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' existing = if (useExisting) {
  name: existingAccountName
  scope: resourceGroup(empty(existingResourceGroup) ? resourceGroup().name : existingResourceGroup)
}

var effectiveAccountName = useExisting ? existingAccountName : accountName

// We can only declare child resources inline against newAccount (existing scope across RG
// cannot be safely managed here). When useExisting=true we still create the database and
// containers idempotently via a nested scope only when the account is in the current RG.
var manageChildren = !useExisting || empty(existingResourceGroup)

resource accountRef 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' existing = if (manageChildren) {
  name: effectiveAccountName
  dependsOn: [
    newAccount
  ]
}

// --- Database --------------------------------------------------------------

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-11-15' = if (manageChildren) {
  parent: accountRef
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
  }
}

// --- Containers ------------------------------------------------------------

resource memoriesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = if (manageChildren) {
  parent: database
  name: 'memories'
  properties: {
    resource: {
      id: 'memories'
      defaultTtl: -1
      partitionKey: {
        kind: 'MultiHash'
        version: 2
        paths: [
          '/user_id'
          '/thread_id'
        ]
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/embedding/?'
          }
          {
            path: '/source_memory_ids/*'
          }
          {
            path: '/supersedes_ids/*'
          }
          {
            path: '/"_etag"/?'
          }
        ]
        vectorIndexes: [
          {
            path: '/embedding'
            type: 'diskANN'
          }
        ]
      }
      vectorEmbeddingPolicy: {
        vectorEmbeddings: [
          {
            path: '/embedding'
            dataType: 'float32'
            distanceFunction: 'cosine'
            dimensions: 1536
          }
        ]
      }
      fullTextPolicy: {
        defaultLanguage: 'en-US'
        fullTextPaths: [
          {
            path: '/content'
            language: 'en-US'
          }
        ]
      }
    }
  }
}

resource leasesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = if (manageChildren && deployFunctionContainers) {
  parent: database
  name: 'leases'
  properties: {
    resource: {
      id: 'leases'
      partitionKey: {
        kind: 'Hash'
        paths: [
          '/id'
        ]
      }
    }
  }
}

resource counterContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = if (manageChildren && deployFunctionContainers) {
  parent: database
  name: 'counter'
  properties: {
    resource: {
      id: 'counter'
      partitionKey: {
        kind: 'MultiHash'
        version: 2
        paths: [
          '/user_id'
          '/thread_id'
        ]
      }
    }
  }
}

// --- Outputs ---------------------------------------------------------------

output accountName string = effectiveAccountName
output endpoint string = useExisting ? existingAccount!.properties.documentEndpoint : newAccount!.properties.documentEndpoint
output databaseName string = databaseName
output memoriesContainerName string = 'memories'
output leasesContainerName string = 'leases'
output counterContainerName string = 'counter'
output accountResourceId string = useExisting ? existingAccount!.id : newAccount!.id
