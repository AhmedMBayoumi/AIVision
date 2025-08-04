plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("gemma_model_pack")
    dynamicDelivery {
        deliveryType.set("install-time")
    }
}