// Cosmos DB NoSQL serverless account + database + containers for the Agent
// Memory Toolkit. Single-file module — no `existing` keyword tricks because
// the account is always created fresh by this template.

@description('Name of the Cosmos account to create.')
param accountName string

@description('Azure region.')
param location string

@description('Tags to apply to the account.')
param tags object = {}

@description('Database name. Created if missing.')
param databaseName string = 'ai_memory'

@description('Turns container name.')
param turnsContainerName string = 'memories_turns'

@description('Default TTL for turn documents, in seconds. Use -1 to disable expiry.')
param memoriesTurnsDefaultTtl int = 2592000

@description('Whether to also create the Durable Function support containers (leases, counter).')
param deployFunctionContainers bool = true

@description('Vector embedding output dimensions. Wired through from main.bicep so both Cosmos and the function app stay in sync.')
param embeddingDimensions int = 1536

// --- Account --------------------------------------------------------------

resource account 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = {
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
      {
        name: 'EnableNoSQLVectorSearch'
      }
      {
        name: 'EnableNoSQLFullTextSearch'
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    disableLocalAuth: false
    publicNetworkAccess: 'Enabled'
  }
}

// --- Database -------------------------------------------------------------

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-11-15' = {
  parent: account
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
  }
}

// --- Containers -----------------------------------------------------------

resource memoriesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
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
            dimensions: embeddingDimensions
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

resource memoriesTurnsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  parent: database
  name: turnsContainerName
  properties: {
    resource: {
      id: turnsContainerName
      defaultTtl: memoriesTurnsDefaultTtl
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
            dimensions: embeddingDimensions
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

resource leasesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = if (deployFunctionContainers) {
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

resource counterContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = if (deployFunctionContainers) {
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
      defaultTtl: 7776000
    }
  }
}

// --- Outputs --------------------------------------------------------------

output accountName string = account.name
output accountResourceId string = account.id
output endpoint string = account.properties.documentEndpoint
output databaseName string = databaseName
output memoriesContainerName string = 'memories'
output turnsContainerName string = turnsContainerName
output leasesContainerName string = 'leases'
output counterContainerName string = 'counter'
