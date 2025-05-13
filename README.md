Simple example of building a Chat barebones chat with an LLM interface using Modus and DGraph running on Hypermode.

If you would like to use this code, make sure to create a .env.local file and add 2 env variables
MODUS_MODEL_ROUTER_HYP_WKS_KEY=""
MODUS_WEBSITE_HYP_DGRAPH_KEY=""

The fist is for model router and the second for your hosted DGraph database.

Update the connString on modus.json for the "website" entry to your actual DGraph DB.
