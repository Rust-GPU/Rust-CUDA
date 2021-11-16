use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::*;

#[proc_macro_attribute]
pub fn trace_ffi_calls(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as ItemForeignMod);
    let clone = item.clone();
    let priv_module: ItemMod = parse_quote! {
        mod private {
            use super::*;
            #clone
        }
    };
    let mut module: ItemMod = parse_quote!(
        pub(crate) mod public {
            use super::*;
        }
    );

    for foreign in item.items {
        if let ForeignItem::Fn(func) = foreign {
            let contents = &mut module.content.as_mut().unwrap().1;
            let Signature {
                ident,
                generics,
                inputs,
                output,
                ..
            } = &func.sig;

            let args = inputs
                .into_iter()
                .map(|arg| match arg {
                    FnArg::Typed(ty) => (*ty.pat).clone(),
                    _ => unreachable!(),
                })
                .collect::<Punctuated<Pat, Comma>>();

            let new_func = parse_quote! {
                pub(crate) unsafe fn #ident #generics(#inputs) #output {
                    tracing::trace!(stringify!(#ident));
                    super::private::#ident(#args)
                }
            };

            contents.push(Item::Fn(new_func));
        }
    }

    let tokens = quote! {
        #priv_module
        #module
        pub(crate) use public::*;
    };

    tokens.into()
}
